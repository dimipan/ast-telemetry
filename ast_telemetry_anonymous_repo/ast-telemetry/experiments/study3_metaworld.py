import json

import numpy as np

from asttel import events, trajectory
from asttel.bindings import metaworld as mw
from asttel.data import output_dir, write_json

N_SEEDS = 20
WARMUP = 5
SATURATED = 3.95


def save(out, name, seed, records):
    trajectory.write(out / "trajectories" / f"{name}_s{seed}.jsonl",
                     {"study": "metaworld", "condition": name, "seed": seed, "schema": mw.SCHEMA}, records)


def main():
    out = output_dir("study3_metaworld")
    seeds = mw.qualify(range(N_SEEDS))
    clean, i1, i2, onset_coverage = [], [], [], []
    for seed, sign in sorted(seeds.items()):
        c, _ = mw.run_session(seed, sign)
        r1, _ = mw.run_session(seed, sign, mw.I1_SPANS, "orbit")
        clean.append(c)
        i1.append(r1)
        save(out, "C0", seed, c)
        save(out, "I1", seed, r1)
        t_success = next((r["t"] for r in c if r["engine_success"]), None)
        if t_success is None or t_success + mw.I2_MARGIN + mw.I2_WIDTH > mw.MAX_EXCHANGES:
            continue
        a = t_success + mw.I2_MARGIN
        r2, _ = mw.run_session(seed, sign, ((a, a + mw.I2_WIDTH),), "loiter")
        i2.append(r2)
        save(out, "I2", seed, r2)
        onset_coverage.append(sum(next(r for r in c if r["t"] == a - 1)["upsilon"].values()))
    first = min(seeds)
    save(out, "random", first, mw.run_session(first, seeds[first], random_policy=True)[0])
    result = {
        "qualified_seeds": sorted(seeds),
        "I1": events.si_metrics(i1, WARMUP),
        "I2": events.si_metrics(i2, WARMUP),
        "I2_zero_gain_fraction": events.zero_gain_fraction(i2),
        "I2_saturated_at_onset": float(np.mean([c >= SATURATED for c in onset_coverage])),
        "pe_clean": events.pe_metrics(clean, WARMUP),
        "pe_injected": events.pe_metrics(i1, WARMUP),
    }
    write_json(out / "summary.json", result)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
