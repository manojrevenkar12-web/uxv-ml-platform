"""Distributed, reproducible training with experiment tracking.

* PyTorch DistributedDataParallel. Uses NCCL on GPUs and Gloo on CPU. Works under
  `torchrun` (Kubernetes / multi-node: env:// rendezvous) or spawns local
  ranks itself.
* Every run is pinned to an immutable dataset version; the version id, the
  manifest and the code version are logged with the run, and the final weights
  are hashed so two runs can be compared bit for bit.
* Tracking via MLflow (any backend: local SQLite, a tracking server, ...).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import socket
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from ..data.dataset import load_index, load_manifest
from ..data.loader import ShardStream, collate
from ..data.store import open_store
from .model import CenterNetLite
from .targets import decode, detection_loss, make_targets, map_from_records, per_class_records


@dataclass
class TrainConfig:
    dataset: str = "uxv"
    dataset_version: str = "latest"
    inputs: str = "rgb+lidar"           # "rgb" or "rgb+lidar"
    epochs: int = 14
    batch_size: int = 32                # global batch (split across ranks)
    lr: float = 3e-3
    weight_decay: float = 1e-4
    width: int = 32
    seed: int = 0
    world_size: int = 2
    threads_per_rank: int = 1
    train_flights: list | None = None   # restrict to these flights (None = all train flights)
    exclude_vendors: list = field(default_factory=list)
    label_override: str | None = None   # store ref with replacement labels (e.g. auto-labels)
    experiment: str = "uxv-perception"
    run_name: str = "train"
    register_as: str | None = "uxv-detector"


def in_channels(inputs: str) -> int:
    return 5 if "lidar" in inputs else 3


def code_version() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=Path(__file__).parent, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return os.environ.get("UXVML_CODE_VERSION", "unknown")


def weights_digest(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def train_subset(store, manifest, cfg: TrainConfig):
    idx = load_index(store, manifest)
    tr = idx[idx.split == "train"]
    if cfg.train_flights is not None:
        tr = tr[tr.flight_id.isin(cfg.train_flights)]
    if cfg.exclude_vendors:
        tr = tr[~tr.vendor.isin(cfg.exclude_vendors)]
    return set(tr.sample_id)


def load_label_override(store, ref):
    if not ref:
        return None
    doc = store.get_ref(ref)
    return {sid: np.array(b, np.float32).reshape(-1, 5) for sid, b in doc["labels"].items()}


@torch.no_grad()
def predict(model, store, manifest, split, inputs, batch_size=64, include=None):
    model.eval()
    ds = ShardStream(store, manifest, split, inputs=inputs, train=False, include=include)
    preds, gts, sids = [], [], []
    for batch in DataLoader(ds, batch_size=batch_size, collate_fn=collate):
        dev = next(model.parameters()).device
        heat, reg = model(batch["x"].to(dev).contiguous(memory_format=torch.channels_last))
        preds += decode(heat, reg)
        gts += batch["boxes"]
        sids += batch["sid"]
    return sids, preds, gts


def _worker(rank: int, world: int, cfg: TrainConfig, store_uri: str, tracking_uri: str,
            out_dir: str, init: str | None):
    torch.set_num_threads(cfg.threads_per_rank)
    if world > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if init:
            dist.init_process_group(backend, init_method=init, rank=rank, world_size=world)
        else:
            dist.init_process_group(backend)  # torchrun / env://
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    store = open_store(store_uri)
    manifest = load_manifest(store, cfg.dataset, cfg.dataset_version)
    include = train_subset(store, manifest, cfg)
    override = load_label_override(store, cfg.label_override)
    n_train = len(include)
    per_rank = cfg.batch_size // world
    steps = max(1, n_train // cfg.batch_size)

    model = CenterNetLite(in_channels(cfg.inputs), cfg.width).to(device, memory_format=torch.channels_last)
    ddp = DDP(model) if world > 1 else model
    opt = torch.optim.AdamW(ddp.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total = steps * cfg.epochs
    warm = max(1, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))

    ds = ShardStream(store, manifest, "train", inputs=cfg.inputs, train=True, seed=cfg.seed,
                     rank=rank, world=world, include=include, label_override=override,
                     target_fn=make_targets)
    it = iter(DataLoader(ds, batch_size=per_rank, collate_fn=collate, num_workers=0))

    mlflow = None
    if rank == 0:
        import mlflow  # noqa: F811
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(cfg.experiment)
        mlflow.start_run(run_name=cfg.run_name)
        mlflow.log_params({k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
                           for k, v in asdict(cfg).items()})
        mlflow.log_params({"dataset_version_resolved": manifest["version"],
                           "dataset_index_digest": manifest["index"], "code_version": code_version(),
                           "n_train_samples": n_train, "steps_per_epoch": steps,
                           "device": str(device), "backend": "nccl" if torch.cuda.is_available() else ("gloo" if world > 1 else "none")})
        mlflow.set_tags({"dataset": f"{cfg.dataset}@{manifest['version']}", "inputs": cfg.inputs})

    t0 = time.time()
    seen = 0
    for epoch in range(cfg.epochs):
        ddp.train()
        losses = []
        te = time.time()
        for _ in range(steps):
            b = next(it)
            x = b["x"].to(device).contiguous(memory_format=torch.channels_last)
            heat, reg = ddp(x)
            loss, parts = detection_loss(heat, reg, b["heat"].to(device), b["reg"].to(device), b["mask"].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddp.parameters(), 5.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
            seen += x.shape[0] * world
        if rank == 0:
            msg = {"train_loss": float(np.mean(losses)), "epoch_seconds": time.time() - te,
                   "samples_per_second": steps * cfg.batch_size / (time.time() - te)}
            if epoch == cfg.epochs - 1 or epoch % 4 == 3:
                _, p, g = predict(model, store, manifest, "val", cfg.inputs)
                _, m = map_from_records(per_class_records(p, g))
                msg["val_mAP"] = m
            mlflow.log_metrics(msg, step=epoch)
            print(f"[rank0] epoch {epoch + 1}/{cfg.epochs} " + " ".join(f"{k}={v:.4f}" for k, v in msg.items()), flush=True)

    if world > 1:
        dist.barrier()
    result = None
    if rank == 0:
        wd = weights_digest(model)
        aps, m = map_from_records(per_class_records(*predict(model, store, manifest, "val", cfg.inputs)[1:]))
        mlflow.log_metrics({"final_val_mAP": m, **{f"val_AP_{c}": v for c, v in aps.items()},
                            "train_seconds": time.time() - t0})
        mlflow.set_tag("weights_sha256", wd)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out / "model.pt")
        card = {"config": asdict(cfg), "dataset": cfg.dataset, "dataset_version": manifest["version"],
                "dataset_index": manifest["index"], "raw_inputs": manifest["inputs"],
                "code_version": code_version(), "weights_sha256": wd,
                "in_channels": in_channels(cfg.inputs), "width": cfg.width,
                "val_mAP": m, "world_size": world}
        (out / "model_card.json").write_text(json.dumps(card, indent=2))
        mlflow.log_artifact(str(out / "model.pt"), "model")
        mlflow.log_artifact(str(out / "model_card.json"), "model")
        run = mlflow.active_run()
        version = None
        if cfg.register_as:
            from mlflow import MlflowClient
            client = MlflowClient()
            try:
                client.create_registered_model(cfg.register_as)
            except Exception:
                pass
            mv = client.create_model_version(cfg.register_as, f"{run.info.artifact_uri}/model", run.info.run_id,
                                             tags={"dataset_version": manifest["version"], "weights_sha256": wd})
            version = mv.version
        card["mlflow_run_id"] = run.info.run_id
        card["registered_version"] = version
        (out / "model_card.json").write_text(json.dumps(card, indent=2))
        mlflow.end_run()
        result = card
    if world > 1:
        dist.destroy_process_group()
    return result


def _spawn_entry(rank, world, cfg, store_uri, tracking_uri, out_dir, init):
    _worker(rank, world, cfg, store_uri, tracking_uri, out_dir, init)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def train(cfg: TrainConfig, store_uri: str, tracking_uri: str, out_dir: str) -> dict:
    """Runs training; returns the model card. Uses torchrun's env if present."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        res = _worker(rank, world, cfg, store_uri, tracking_uri, out_dir, None)
        return res or {}
    world = cfg.world_size
    if world == 1:
        return _worker(0, 1, cfg, store_uri, tracking_uri, out_dir, None)
    init = f"file://{tempfile.mkdtemp()}/rdzv"
    torch.multiprocessing.spawn(_spawn_entry, args=(world, cfg, store_uri, tracking_uri, out_dir, init),
                                nprocs=world, join=True)
    return json.loads((Path(out_dir) / "model_card.json").read_text())


def load_model(model_dir: str | Path):
    model_dir = Path(model_dir)
    card = json.loads((model_dir / "model_card.json").read_text())
    m = CenterNetLite(card["in_channels"], card["width"])
    m.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
    m.eval()
    return m, card
