import collections

import crafter
import crafter.engine
import numpy as np

from asttel.monitor import Monitor


class OrderedSet(dict):
    def add(self, item):
        self[item] = None

    def remove(self, item):
        del self[item]


_world_reset = crafter.engine.World.reset


def _deterministic_reset(self, seed=None):
    _world_reset(self, seed)
    self._chunks = collections.defaultdict(OrderedSet)


crafter.engine.World.reset = _deterministic_reset

K_STEPS = 25
INVENTORY = sorted(["health", "food", "drink", "energy", "sapling", "wood", "stone", "coal", "iron", "diamond",
                    "wood_pickaxe", "stone_pickaxe", "iron_pickaxe", "wood_sword", "stone_sword", "iron_sword"])


def has(v, k):
    return v.get(k, 0) >= 1


def done(name):
    return lambda v, a: a.get(name, 0) > 0


RUBRIC = {
    "collect_wood": [lambda v, a: has(v, "wood")],
    "place_table": [lambda v, a: has(v, "wood"), done("place_table")],
    "make_wood_pickaxe": [lambda v, a: has(v, "wood"), done("place_table"), done("make_wood_pickaxe")],
    "make_wood_sword": [lambda v, a: has(v, "wood"), done("place_table"), done("make_wood_sword")],
    "collect_stone": [done("make_wood_pickaxe"), done("collect_stone")],
    "place_stone": [lambda v, a: has(v, "stone"), done("place_stone")],
    "make_stone_pickaxe": [lambda v, a: has(v, "stone"), done("place_table"), done("make_stone_pickaxe")],
    "make_stone_sword": [lambda v, a: has(v, "stone"), done("place_table"), done("make_stone_sword")],
    "collect_coal": [done("make_wood_pickaxe"), done("collect_coal")],
    "collect_iron": [done("make_stone_pickaxe"), done("collect_iron")],
    "place_furnace": [lambda v, a: has(v, "stone"), done("place_furnace")],
    "make_iron_pickaxe": [done("place_furnace"), lambda v, a: has(v, "iron"), done("make_iron_pickaxe")],
    "make_iron_sword": [done("place_furnace"), lambda v, a: has(v, "iron"), done("make_iron_sword")],
    "collect_diamond": [done("make_iron_pickaxe"), done("collect_diamond")],
    **{name: [done(name)] for name in ["collect_sapling", "place_plant", "eat_plant", "eat_cow", "collect_drink",
                                        "defeat_zombie", "defeat_skeleton", "wake_up"]},
}
SCHEMA = list(RUBRIC)


def score(inventory, achievements):
    return {name: sum(p(inventory, achievements) for p in preds) / len(preds) for name, preds in RUBRIC.items()}


class ActivePolicy:
    def __init__(self, names):
        weights = np.array([3.0 if n.startswith("move_") else 4.0 if n == "do" else 0.45 if n.startswith("place_")
                            else 0.35 if n.startswith("make_") else 0.05 if n in {"noop", "sleep"} else 0.15
                            for n in names])
        self.probs = weights / weights.sum()

    def __call__(self, rng):
        return int(rng.choice(len(self.probs), p=self.probs))


def run_session(seed, stall_spans=(), max_exchanges=60):
    env = crafter.Env(seed=seed)
    env.reset()
    rng = np.random.default_rng(seed)
    names = env.action_names
    active, noop = ActivePolicy(names), names.index("noop")
    monitor = Monitor(SCHEMA)
    records = []
    for t in range(1, max_exchanges + 1):
        stall = any(a <= t < b for a, b in stall_spans)
        actions = np.zeros(len(names))
        start = np.array(env._player.pos, dtype=float)
        for _ in range(K_STEPS):
            a = noop if stall else active(rng)
            actions[a] += 1
            _, _, finished, info = env.step(a)
            if finished:
                break
        inv, ach = info["inventory"], info["achievements"]
        embedding = np.concatenate([actions, [inv.get(k, 0) for k in INVENTORY],
                                    [ach.get(s, 0) > 0 for s in SCHEMA],
                                    np.array(env._player.pos, dtype=float) - start]).astype(float)
        scores = score(inv, ach)
        delta = {d: scores[d] - monitor.upsilon[d] for d in SCHEMA}
        target = max(delta, key=delta.get)
        if delta[target] <= 0:
            target = max(SCHEMA, key=lambda d: scores[d] * (1.0 - monitor.upsilon[d]))
        pe, si, gains = monitor.step(embedding, scores, target)
        records.append({"t": t, "target": target, "si": si, "pe": pe, "gain": sum(gains.values()),
                        "upsilon": dict(monitor.upsilon), "stall_gt": stall})
        if finished:
            break
    return records
