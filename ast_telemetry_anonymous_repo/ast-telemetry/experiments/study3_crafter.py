import json

import numpy as np

from asttel import events, trajectory
from asttel.bindings import crafter as cr
from asttel.data import output_dir, write_json

N_SEEDS = 100
WARMUP = 2
SPANS = ((3, 6), (8, 11))


def atomicity(runs):
    gains = np.array([r["gain"] for records in runs for r in records])
    positive = gains[gains > 1e-9]
    return {"positive_gain_fraction": float(np.mean(gains > 1e-9)),
            "atomic_fraction_of_positive_updates": float(np.mean(positive >= 1.0))}


def main():
    out = output_dir("study3_crafter")
    clean = [cr.run_session(s) for s in range(N_SEEDS)]
    injected = [cr.run_session(s, SPANS) for s in range(N_SEEDS)]
    for name, runs in (("clean", clean), ("injected", injected)):
        for seed, records in enumerate(runs):
            trajectory.write(out / "trajectories" / f"{name}_s{seed}.jsonl",
                             {"study": "crafter", "condition": name, "seed": seed, "schema": cr.SCHEMA}, records)
    result = {
        "injected": events.si_metrics(injected, WARMUP, close_open_spans=True),
        "atomicity_clean": atomicity(clean),
        "pe_clean": events.pe_metrics(clean, WARMUP),
        "pe_injected": events.pe_metrics(injected, WARMUP),
    }
    write_json(out / "summary.json", result)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
