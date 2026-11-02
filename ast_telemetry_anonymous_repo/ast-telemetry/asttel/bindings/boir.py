import re
from pathlib import Path

import numpy as np

from asttel.monitor import Monitor

DELTA, W_ANGLE, W_PATH = 0.2, 0.6, 0.4
LIMIT_ANGLE, LIMIT_PATH = 180.0, 25.0
K = 5


def trials(scenario_dir):
    return sorted(m.group(1) for f in (Path(scenario_dir) / "path").glob("path1_*.text")
                  if (m := re.match(r"^path1_(.+)\.text$", f.name)))


def load(scenario_dir, trial):
    d = Path(scenario_dir)
    n = max(int(re.match(r"^path(\d+)_", f.name).group(1)) for f in (d / "path").glob(f"path*_{trial}.text"))
    cols = {kind: [np.loadtxt(d / kind / f"{kind}{i}_{trial}.text", delimiter=",", usecols=(1,))
                   for i in range(1, n + 1)] for kind in ("path", "angle")}
    length = min(len(c) for cs in cols.values() for c in cs)
    paths = np.stack([c[:length] for c in cols["path"]], axis=1)
    angles = np.stack([c[:length] for c in cols["angle"]], axis=1)
    clock = np.loadtxt(d / "path" / f"path1_{trial}.text", delimiter=",", usecols=(0,))[:length]
    return paths, angles, clock


def epochs(scenario_dir, trial, clock):
    f = Path(scenario_dir) / "change" / f"change_{trial}.text"
    changes = np.loadtxt(f, delimiter=",", usecols=(0,)) if f.exists() else np.array([])
    if np.size(changes) == 0:
        return [(0, len(clock))]
    cuts = [int(np.argmin(np.abs(clock - c))) for c in np.sort(np.atleast_1d(changes))]
    bounds = [0, *cuts, len(clock)]
    return list(zip(bounds[:-1], bounds[1:]))


def posteriors(angles, paths):
    n = angles.shape[1]
    cpt = np.ones((n, n)) * (DELTA / (n - 1))
    np.fill_diagonal(cpt, 1 - DELTA)
    post, out = np.ones(n) / n, np.zeros(angles.shape)
    for t in range(len(angles)):
        like = np.exp(-(angles[t] / LIMIT_ANGLE) / W_ANGLE) * np.exp(-(paths[t] / LIMIT_PATH) / W_PATH)
        post = like * (cpt @ post)
        post = post / post.sum()
        out[t] = post
    return out


def embed(post, angles, map_idx):
    entropy = -(post * np.log2(np.clip(post, 1e-12, 1))).sum(axis=1)
    kl = np.array([np.sum(post[t] * np.log2(np.clip(post[t], 1e-12, 1) / np.clip(post[t - 1], 1e-12, 1)))
                   for t in range(1, len(post))])
    switches = float(np.sum(np.diff(post.argmax(axis=1)) != 0))
    dang = np.diff(angles[:, map_idx])
    reversals = float(np.sum(np.diff(np.sign(dang[np.abs(dang) > 0.5])) != 0)) if len(dang) > 1 else 0.0
    e = np.concatenate([post.mean(axis=0), post[-1] - post[0], [entropy.mean(), entropy[-1] - entropy[0]],
                        [kl.mean() if len(kl) else 0.0, switches],
                        [np.abs(dang).mean() if len(dang) else 0.0, reversals]])
    norm = np.linalg.norm(e)
    return e / norm if norm > 1e-12 else e


def run_trial(scenario_dir, trial, true_goals):
    paths, angles, clock = load(scenario_dir, trial)
    P = posteriors(angles, paths)
    n = P.shape[1]
    hyps = [f"h_goal_{i + 1}" for i in range(n)]
    records, t = [], 0
    for ep, (a, b) in enumerate(epochs(scenario_dir, trial, clock)):
        monitor = Monitor(hyps, aggregate_gain=True)
        true_idx = true_goals[ep] if ep < len(true_goals) else None
        for w0 in range(a, b - K + 1, K):
            window = slice(w0, w0 + K)
            map_idx = int(np.bincount(P[window].argmax(axis=1), minlength=n).argmax())
            p_end = P[w0 + K - 1]
            if true_idx is None:
                p = np.clip(p_end, 1e-9, 1 - 1e-9)
                s = 1.0 + p * np.log2(p) + (1 - p) * np.log2(1 - p)
            else:
                s = np.array([p_end[i] if i == true_idx else 1.0 - p_end[i] for i in range(n)])
            scores = {h: float(np.clip(s[i], 0, 1)) for i, h in enumerate(hyps)}
            pe, si, gains = monitor.step(embed(P[window], angles[window], map_idx), scores, hyps[map_idx])
            t += 1
            records.append({"t": t, "epoch": ep, "target": hyps[map_idx], "si": si, "pe": pe,
                            "gain": sum(gains.values()), "upsilon": dict(monitor.upsilon),
                            "posterior": {h: float(p_end[i]) for i, h in enumerate(hyps)},
                            "true_goal": hyps[true_goals[min(ep, len(true_goals) - 1)]]})
    return records
