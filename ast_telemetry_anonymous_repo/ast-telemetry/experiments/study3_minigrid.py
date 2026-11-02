import json

from asttel import events, trajectory
from asttel.bindings import minigrid
from asttel.data import output_dir, write_json

N_SEEDS = 20
WARMUP = 5
I1_SPANS = ((8, 20), (40, 50))
I2_SPANS = ((55, 67),)


def save(out, name, runs):
    for seed, records in enumerate(runs):
        trajectory.write(out / "trajectories" / f"{name}_s{seed}.jsonl",
                         {"study": "minigrid", "condition": name, "seed": seed, "schema": minigrid.SCHEMA}, records)


def main():
    out = output_dir("study3_minigrid")
    i1 = [minigrid.run_session(s, I1_SPANS) for s in range(N_SEEDS)]
    i2 = [minigrid.run_session(s, I2_SPANS) for s in range(N_SEEDS)]
    save(out, "I1", i1)
    save(out, "I2", i2)
    result = {
        "I1": events.si_metrics(i1, WARMUP),
        "I2": events.si_metrics(i2, WARMUP),
        "I2_zero_gain_fraction": events.zero_gain_fraction(i2),
        "pe": events.pe_metrics(i1, WARMUP),
    }
    write_json(out / "summary.json", result)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
