import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
from minigrid.wrappers import FullyObsWrapper

from asttel.monitor import Monitor

K_STEPS = 8
SECTORS = 3
SCHEMA = [f"sector_{r}_{c}" for r in range(SECTORS) for c in range(SECTORS)]
DIRS = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
LOOP = [2, 1, 2, 1, 2, 1, 2, 1]


def sector_of(x, y, w, h):
    return f"sector_{min(SECTORS - 1, y * SECTORS // h)}_{min(SECTORS - 1, x * SECTORS // w)}"


def forward_cell(env):
    x, y = env.unwrapped.agent_pos
    dx, dy = DIRS[env.unwrapped.agent_dir]
    return x + dx, y + dy


class Explorer:
    def __init__(self, rng):
        self.rng = rng

    def act(self, env, consumed):
        fx, fy = forward_cell(env)
        cell = env.unwrapped.grid.get(fx, fy)
        if cell is not None and cell.type == "wall":
            return self.rng.choice([0, 1])
        p_forward = 0.85 if not consumed[fx, fy] else 0.6
        return 2 if self.rng.random() < p_forward else self.rng.choice([0, 1])


class Looper:
    def __init__(self):
        self.k = 0

    def act(self, env, consumed):
        a = LOOP[self.k % len(LOOP)]
        self.k += 1
        return a


def run_session(seed, stall_spans, n_exchanges=80):
    env = FullyObsWrapper(gym.make("MiniGrid-FourRooms-v0"))
    env.reset(seed=seed)
    grid = env.unwrapped.grid
    w, h = grid.width, grid.height
    walkable = np.array([[(c := grid.get(x, y)) is None or c.type in ("goal", "door") for y in range(h)]
                         for x in range(w)])
    cells = {s: [(x, y) for x in range(w) for y in range(h) if walkable[x, y] and sector_of(x, y, w, h) == s]
             for s in SCHEMA}
    monitor = Monitor(SCHEMA)
    consumed = np.zeros((w, h), dtype=bool)
    explorer, looper = Explorer(np.random.default_rng(seed)), Looper()
    records = []
    for t in range(1, n_exchanges + 1):
        stall = any(a <= t < b for a, b in stall_spans)
        policy = looper if stall else explorer
        visits, actions = np.zeros((w, h)), np.zeros(3)
        steps = {s: 0 for s in SCHEMA}
        for _ in range(K_STEPS):
            x, y = env.unwrapped.agent_pos
            consumed[x, y] = True
            fx, fy = forward_cell(env)
            if 0 <= fx < w and 0 <= fy < h and walkable[fx, fy]:
                consumed[fx, fy] = True
            visits[x, y] += 1
            steps[sector_of(x, y, w, h)] += 1
            a = policy.act(env, consumed)
            actions[a] += 1
            env.step(a)
        embedding = np.concatenate([visits.flatten(), actions])
        scores = {s: float(sum(consumed[x, y] for x, y in cells[s])) / max(1, len(cells[s])) for s in SCHEMA}
        target = max(steps, key=steps.get)
        pe, si, gains = monitor.step(embedding, scores, target)
        records.append({"t": t, "target": target, "si": si, "pe": pe, "gain": sum(gains.values()),
                        "upsilon": dict(monitor.upsilon), "stall_gt": stall})
    env.close()
    return records
