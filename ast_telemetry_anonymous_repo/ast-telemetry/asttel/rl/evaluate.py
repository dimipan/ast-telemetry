import numpy as np

from asttel.params import THETA_SI


def rollout(model, env, seed, records=None):
    obs, _ = env.reset(seed=seed)
    si, ret, done = [], 0.0, False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        si.append(info["SI"])
        ret += float(reward)
        done = terminated or truncated
        if records is not None:
            records.append({"t": len(si), "target": info["target"], "upsilon": info["upsilon"], "pe": info["PE"],
                            "si": info["SI"]})
    n = len(env.dims)
    return {
        "categories_resolved": info["complete_dimensions"],
        "total_knowledge": info["total_knowledge"],
        "full_resolution": float(info["complete_dimensions"] == n),
        "mean_si": float(np.mean(si)),
        "stall_burden": 100.0 * float(np.mean(np.array(si) > THETA_SI)),
        "turns": len(si),
        "truncated": float(truncated),
        "return": ret,
    }


def bootstrap_ci(values, n_boot=10000, alpha=0.05, seed=0):
    x = np.asarray(values, dtype=float)
    boots = x[np.random.default_rng(seed).integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(x.mean()), float(lo), float(hi)
