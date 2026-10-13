from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .flight import FlightSpec, generate_flight, plan_flights


def _gen(args):
    spec, raw, truth = args
    generate_flight(spec, Path(raw), Path(truth))
    return spec.flight_id


def simulate(out_dir: str | Path, n_flights: int, seed: int = 0, duration_s: float = 32.0,
             workers: int = 2) -> list[FlightSpec]:
    """Generates raw flight logs into out_dir/raw and hidden truth into out_dir/truth."""
    out = Path(out_dir)
    specs = plan_flights(n_flights, seed, duration_s)
    with ProcessPoolExecutor(workers) as ex:
        list(ex.map(_gen, [(s, out / "raw", out / "truth") for s in specs]))
    return specs
