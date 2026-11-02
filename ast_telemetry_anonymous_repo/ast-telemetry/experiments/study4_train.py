import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor as EpisodeLog
from stable_baselines3.common.utils import set_random_seed

from asttel.data import load_embeddings, load_json, output_dir
from asttel.rl.env import VARIANTS, InterviewEnv

CORPORA = ["child", "elderly", "hiker"]
PPO_ARGS = dict(n_steps=256, batch_size=64, n_epochs=10, learning_rate=3e-4, ent_coef=0.02, device="cpu")
CALIBRATION_EPISODES = 200
CALIBRATION_SEED = 10000


def corpus(name):
    return {"name": name, **load_json("sar", f"{name}.json")}


def train(name, variant, condition, seed, steps, rho, out):
    torch.set_num_threads(1)
    set_random_seed(seed)
    c, embeddings = corpus(name), load_embeddings("sar")
    path = Path(out) / f"ppo_{variant}_cond{condition}_seed{seed}_{name}"
    env = EpisodeLog(InterviewEnv(c, embeddings, variant, condition, rho=rho), str(path) + ".monitor.csv")
    env.reset(seed=seed)
    model = PPO("MlpPolicy", env, seed=seed, verbose=0, **PPO_ARGS)
    model.learn(total_timesteps=steps)
    model.save(path)
    return str(path) + ".zip"


def calibrate():
    friction, operational = [], []
    for name in CORPORA:
        env = InterviewEnv(corpus(name), load_embeddings("sar"), "Coupled-Blind", "A")
        rng = np.random.default_rng(0)
        for ep in range(CALIBRATION_EPISODES):
            env.reset(seed=CALIBRATION_SEED + ep)
            done = False
            while not done:
                _, _, terminated, truncated, info = env.step(int(rng.integers(env.action_space.n)))
                friction.append(info["SI"])
                operational.append(info["operational_cost"])
                done = terminated or truncated
    return float(np.sum(friction) / np.sum(operational)), {
        "reference_policy": "uniform random", "episodes_per_corpus": CALIBRATION_EPISODES,
        "friction_per_exchange": float(np.mean(friction)), "operational_cost_per_exchange": float(np.mean(operational))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--conditions", nargs="+", default=["A", "B"])
    p.add_argument("--variants", nargs="+", default=list(VARIANTS))
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--rho", type=float, default=None)
    p.add_argument("--jobs", type=int, default=1)
    args = p.parse_args()

    out = Path(args.out) if args.out else output_dir("study4_checkpoints")
    out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.jobs) as pool:
        if args.rho is None:
            rho, detail = calibrate()
            (out / "rho.json").write_text(json.dumps({"rho": rho, **detail}, indent=1))
        else:
            rho = args.rho
        print(f"rho = {rho:.6f}", flush=True)
        grid = list(itertools.product(args.conditions, args.variants, args.seeds, CORPORA))
        jobs = [pool.submit(train, name, variant, condition, seed, args.steps, rho, out)
                for condition, variant, seed, name in grid]
        for job in jobs:
            print(f"trained {Path(job.result()).name}", flush=True)


if __name__ == "__main__":
    main()
