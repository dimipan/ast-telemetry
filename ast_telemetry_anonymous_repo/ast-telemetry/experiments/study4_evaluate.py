import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from asttel import trajectory
from asttel.data import load_embeddings, load_json, output_dir, write_json
from asttel.rl.env import LEGACY_NAMES, VARIANTS, InterviewEnv
from asttel.rl.evaluate import bootstrap_ci, rollout

CORPORA = ["hiker", "child", "elderly"]
METRICS = ["categories_resolved", "total_knowledge", "full_resolution", "mean_si", "stall_burden"]
PATTERN = re.compile(r"^ppo_(?:bayesian_)?(?P<variant>.+)_cond(?P<condition>[AB])_seed(?P<seed>\d+)_(?P<corpus>[a-z]+)\.zip$")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoints")
    p.add_argument("--episodes", type=int, default=1)
    args = p.parse_args()

    out = output_dir("study4")
    embeddings = load_embeddings("sar")
    corpora = {c: load_json("sar", f"{c}.json") for c in CORPORA}
    rows = []
    for path in sorted(Path(args.checkpoints).glob("*.zip")):
        m = PATTERN.match(path.name)
        if not m:
            continue
        variant = LEGACY_NAMES.get(m["variant"], m["variant"])
        seed, corpus, condition = int(m["seed"]), m["corpus"], m["condition"]
        model = PPO.load(path, device="cpu")
        env = InterviewEnv(corpora[corpus], embeddings, variant, condition)
        for ep in range(args.episodes):
            eval_seed = 10000 + seed * 100_000 + CORPORA.index(corpus) * 1_000 + ep
            records = []
            rows.append({"condition": condition, "variant": variant, "seed": seed, "corpus": corpus, "episode": ep,
                         **rollout(model, env, eval_seed, records)})
            if ep == 0:
                trajectory.write(out / "trajectories" / f"{variant}_cond{condition}_seed{seed}_{corpus}.jsonl",
                                 {"study": "rl", "variant": variant, "condition": condition, "seed": seed,
                                  "corpus": corpus, "schema": env.dims}, records)

    with (out / "episodes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    seed_means = defaultdict(list)
    for r in rows:
        seed_means[(r["condition"], r["variant"], r["seed"])].append(r)
    table = {}
    for condition in ("A", "B"):
        for variant in VARIANTS:
            seeds = sorted(s for c, v, s in seed_means if c == condition and v == variant)
            if not seeds:
                continue
            table[f"{condition}/{variant}"] = {
                metric: dict(zip(("mean", "ci_low", "ci_high"), bootstrap_ci(
                    [np.mean([r[metric] for r in seed_means[(condition, variant, s)]]) for s in seeds])))
                for metric in METRICS}
    write_json(out / "table1.json", table)
    for key, row in table.items():
        print(f"{key:18s}", "  ".join(f"{m} {v['mean']:.3f} [{v['ci_low']:.3f}, {v['ci_high']:.3f}]" for m, v in row.items()))


if __name__ == "__main__":
    main()
