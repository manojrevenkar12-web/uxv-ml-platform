"""CI check: a second pipeline run must hit the cache for every step and
resolve to the same dataset version and model weights."""
import json
import sys


def load(path):
    text = open(path).read()
    start = 0 if text.startswith("{") else text.rindex("\n{\n") + 1  # skip log lines before the JSON
    return json.loads(text[start:])


a, b = load(sys.argv[1]), load(sys.argv[2])
assert a["dataset"] == b["dataset"], (a["dataset"], b["dataset"])
assert a["model"]["weights"] == b["model"]["weights"]
slow = {k: v for k, v in b["timings"].items() if v != 0.0}
assert not slow, f"steps re-ran instead of hitting the cache: {slow}"
print(f"cached re-run OK: dataset {a['dataset']['version']}, weights {a['model']['weights'][:16]}, "
      f"decision {b['validation']['decision']}")
