import json

import numpy as np

from asttel import trajectory
from asttel.bindings import boir
from asttel.data import data_path, output_dir, write_json

SCENARIOS = {"S1": ("data1", [1, 0]), "S3": ("data3", [0, 1, 2])}
CONFIDENT = 0.70


def scenario_summary(runs):
    starts = errors = confident = 0
    res_start, res_end, si_post, si_mid = [], [], [], []
    for records in runs:
        for ep in sorted({r["epoch"] for r in records}):
            idx = [i for i, r in enumerate(records) if r["epoch"] == ep]
            first, last = records[idx[0]], records[idx[-1]]
            truth = first["true_goal"]
            starts += 1
            if first["target"] != truth:
                errors += 1
                confident += first["posterior"][first["target"]] > CONFIDENT
            res_start.append(first["upsilon"][truth])
            res_end.append(last["upsilon"][truth])
            if ep > 0 and idx[1:3]:
                si_post.append(np.mean([records[i]["si"] for i in idx[1:3]]))
            si_mid.append(records[idx[len(idx) // 2]]["si"])
    return {"n_trials": len(runs), "epoch_start_map_errors": f"{errors}/{starts}",
            "confident_wrong_starts": f"{confident}/{starts}",
            "true_hypothesis_resolution_start": float(np.mean(res_start)),
            "true_hypothesis_resolution_end": float(np.mean(res_end)),
            "si_post_change": float(np.mean(si_post)), "si_mid_epoch": float(np.mean(si_mid))}


def main():
    out = output_dir("study3_boir")
    result = {}
    for label, (folder, truth) in SCENARIOS.items():
        directory = data_path("boir", folder)
        runs = []
        for trial in boir.trials(directory):
            records = boir.run_trial(directory, trial, truth)
            runs.append(records)
            trajectory.write(out / "trajectories" / f"{label}_{trial}.jsonl",
                             {"study": "boir", "scenario": label, "trial": trial, "schema": list(records[0]["upsilon"])},
                             records)
        result[label] = scenario_summary(runs)
    write_json(out / "summary.json", result)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
