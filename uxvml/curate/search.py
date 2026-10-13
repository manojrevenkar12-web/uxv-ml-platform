"""Dataset search: natural-language queries over generated metadata, plus
query-by-example over learned frame embeddings.

Text is turned into a validated structured query (FrameQuery). The default
parser is deterministic and offline; an LLM parser can be plugged in for
free-form phrasing and must produce the same schema, so a bad generation is
rejected by validation instead of silently returning wrong frames.
"""
from __future__ import annotations

import json
import os
import re
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from ..sim.world import CLASSES

Lighting = Literal["day", "dusk", "night"]
Fog = Literal["none", "light", "heavy"]
Cls = Literal["vehicle", "person", "building"]

CLASS_WORDS = {
    "vehicle": r"vehicles?|cars?|trucks?|vans?",
    "person": r"people|persons?|pedestrians?|humans?",
    "building": r"buildings?|houses?|structures?",
}
NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
             "eight": 8, "nine": 9, "ten": 10}


class FrameQuery(BaseModel):
    lighting: list[Lighting] | None = None
    fog: list[Fog] | None = None
    min_counts: dict[Cls, int] = Field(default_factory=dict)
    max_counts: dict[Cls, int] = Field(default_factory=dict)
    altitude_min_m: float | None = None
    altitude_max_m: float | None = None
    lidar: Literal["any", "present", "missing"] = "any"
    suspect_labels: bool | None = None
    split: list[Literal["train", "val", "test"]] | None = None
    limit: int = Field(50, ge=1, le=10000)


def _num(tok: str) -> int:
    return int(tok) if tok.isdigit() else NUM_WORDS[tok]


def parse_query(text: str) -> FrameQuery:
    t = " " + text.lower().strip() + " "
    q: dict = {"min_counts": {}, "max_counts": {}}

    light = []
    if re.search(r"\b(night|nighttime|dark)\b", t):
        light.append("night")
    if re.search(r"\b(dusk|twilight|evening|dawn|low[- ]light)\b", t):
        light.append("dusk")
    if re.search(r"\b(day|daylight|daytime|bright|sunny)\b", t):
        light.append("day")
    if light:
        q["lighting"] = light

    if re.search(r"\b(heavy|dense|thick) (fog|haze)\b", t):
        q["fog"] = ["heavy"]
    elif re.search(r"\b(light (fog|haze)|hazy|haze|mist|misty)\b", t):
        q["fog"] = ["light"]
    elif re.search(r"\b(no fog|clear|fog[- ]free)\b", t):
        q["fog"] = ["none"]
    elif re.search(r"\b(fog|foggy)\b", t):
        q["fog"] = ["light", "heavy"]

    num = r"(\d+|" + "|".join(NUM_WORDS) + r")"
    for cls, words in CLASS_WORDS.items():
        w = rf"(?:{words})"
        for m in re.finditer(rf"\b(?:at least|min(?:imum)?(?: of)?|>=)\s*{num}\s+(?:\w+\s+)?{w}\b", t):
            q["min_counts"][cls] = _num(m.group(1))
        for m in re.finditer(rf"\b(?:more than|over|>)\s*{num}\s+(?:\w+\s+)?{w}\b", t):
            q["min_counts"][cls] = _num(m.group(1)) + 1
        for m in re.finditer(rf"\b{num}\+\s*{w}\b", t):
            q["min_counts"][cls] = _num(m.group(1))
        for m in re.finditer(rf"\b(?:at most|max(?:imum)?(?: of)?|<=)\s*{num}\s+(?:\w+\s+)?{w}\b", t):
            q["max_counts"][cls] = _num(m.group(1))
        for m in re.finditer(rf"\b(?:fewer than|less than|under|<)\s*{num}\s+(?:\w+\s+)?{w}\b", t):
            q["max_counts"][cls] = max(_num(m.group(1)) - 1, 0)
        if re.search(rf"\b(?:no|without|zero)\s+{w}\b", t):
            q["max_counts"][cls] = 0
        if cls not in q["min_counts"] and cls not in q["max_counts"] and re.search(
                rf"\b(?:with|containing|showing|has|have|and)\s+(?:some\s+|a\s+|an\s+)?{w}\b", t):
            q["min_counts"][cls] = 1
    if re.search(r"\bcrowd(?:s|ed)?\b", t):
        q["min_counts"]["person"] = max(q["min_counts"].get("person", 0), 5)

    unit = r"\s*(?:m|meters?|metres?)\b"
    m = re.search(r"\b(?:below|under|lower than)\s*(\d+(?:\.\d+)?)" + unit, t)
    if m:
        q["altitude_max_m"] = float(m.group(1))
    m = re.search(r"\b(?:above|over|higher than)\s*(\d+(?:\.\d+)?)" + unit, t)
    if m:
        q["altitude_min_m"] = float(m.group(1))
    if re.search(r"\blow(?:[- ]altitude)?\s+(?:altitude|flights?|passes)\b|\blow[- ]altitude\b", t):
        q.setdefault("altitude_max_m", 35.0)
    if re.search(r"\bhigh[- ]altitude\b", t):
        q.setdefault("altitude_min_m", 48.0)

    if re.search(r"\b(missing|no|without|dropped) lidar\b|\blidar (dropout|gap|missing)\b", t):
        q["lidar"] = "missing"
    if re.search(r"\b(suspect|bad|noisy|wrong) labels?\b|\blabel (issues?|errors?|problems?)\b", t):
        q["suspect_labels"] = True
    m = re.search(r"\b(?:in|from) (?:the )?(train|training|val|validation|test)\b", t)
    if m:
        q["split"] = [{"training": "train", "validation": "val"}.get(m.group(1), m.group(1))]
    m = re.search(r"\b(?:top|first|limit)\s+(\d+)\b", t)
    if m:
        q["limit"] = int(m.group(1))
    return FrameQuery(**q)


class LLMQueryParser:
    """Optional: maps free-form text to FrameQuery with a Claude model.

    The model is asked for JSON matching FrameQuery's schema; the answer is
    validated, and anything that fails validation raises instead of running.
    """

    def __init__(self, client=None, model: str | None = None):
        if client is None:
            import anthropic  # optional dependency
            client = anthropic.Anthropic()
        self.client = client
        self.model = model or os.environ.get("UXVML_QUERY_MODEL", "claude-haiku-5-5")

    def __call__(self, text: str) -> FrameQuery:
        schema = json.dumps(FrameQuery.model_json_schema())
        msg = self.client.messages.create(
            model=self.model, max_tokens=400,
            system=("Convert the user's request for drone sensor frames into JSON that validates "
                    f"against this JSON schema. Output only the JSON object.\n{schema}"),
            messages=[{"role": "user", "content": text}],
        )
        raw = msg.content[0].text.strip()
        raw = raw[raw.find("{"): raw.rfind("}") + 1]
        return FrameQuery.model_validate_json(raw)


def run_query(table: pd.DataFrame, q: FrameQuery, count_source: str = "pred") -> pd.DataFrame:
    df = table
    m = np.ones(len(df), bool)
    if q.lighting:
        m &= df["lighting"].isin(q.lighting).values
    if q.fog:
        m &= df["fog"].isin(q.fog).values
    for cls, n in q.min_counts.items():
        m &= (df[f"{count_source}_{cls}"] >= n).values
    for cls, n in q.max_counts.items():
        m &= (df[f"{count_source}_{cls}"] <= n).values
    if q.altitude_min_m is not None:
        m &= (df["altitude_m"] >= q.altitude_min_m).values
    if q.altitude_max_m is not None:
        m &= (df["altitude_m"] <= q.altitude_max_m).values
    if q.lidar != "any":
        m &= (df["lidar_ok"] == (q.lidar == "present")).values
    if q.suspect_labels is not None:
        m &= ((df["suspect_boxes"] > 0) == q.suspect_labels).values
    if q.split:
        m &= df["split"].isin(q.split).values
    return df[m].sort_values("sample_id").head(q.limit)


def similar(emb: np.ndarray, order: list, sample_id: str, k: int = 10, exclude_same_flight: bool = True):
    """Nearest neighbours by cosine similarity of frame embeddings."""
    pos = {s: i for i, s in enumerate(order)}
    q = emb[pos[sample_id]]
    sims = emb @ q / (np.linalg.norm(emb, axis=1) * np.linalg.norm(q) + 1e-9)
    flight = sample_id.split("/")[0]
    out = []
    for i in np.argsort(-sims):
        s = order[i]
        if s == sample_id or (exclude_same_flight and s.split("/")[0] == flight):
            continue
        out.append((s, float(sims[i])))
        if len(out) == k:
            break
    return out


assert set(CLASS_WORDS) == set(CLASSES)
