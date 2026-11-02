from dataclasses import dataclass, field

import metaworld
import numpy as np

from asttel.monitor import Monitor

TASK = "pick-place-v3"
K_STEPS = 10
MAX_EXCHANGES = 50
SCHEMA = ["reach", "grasp", "lift", "place"]
I1_SPANS = ((2, 16),)
I2_WIDTH = 14
I2_MARGIN = 5


def toward(current, target, gain):
    return gain * (np.asarray(target, dtype=np.float64) - np.asarray(current, dtype=np.float64))


@dataclass
class PickPlace:
    close_sign: float
    rng: np.random.Generator
    phase: int = 0
    phase_steps: int = 0
    total_steps: int = field(default=0)

    def act(self, obs, goal, mode=None):
        self.total_steps += 1
        hand, obj = np.asarray(obs[:3], dtype=np.float64), np.asarray(obs[4:7], dtype=np.float64)
        close, opened = float(self.close_sign), float(-self.close_sign)
        a = np.zeros(4, dtype=np.float32)
        grasp_xy = obj[:2] + np.array([-0.005, 0.0])
        hover_z = max(0.20, obj[2] + 0.12)
        grasp_z = obj[2] + 0.025
        lift_z = max(0.18, obj[2] + 0.12)
        goal_hover_z = max(lift_z, goal[2] + 0.08)
        if mode == "orbit":
            angle = self.total_steps * 0.15
            target = np.array([grasp_xy[0] + 0.04 * np.cos(angle), grasp_xy[1] + 0.04 * np.sin(angle), hover_z])
            a[:3] = toward(hand, target, 10.0) + self.rng.normal(0, 0.2, 3)
            a[3] = opened
            return np.clip(a, -1.0, 1.0)
        if mode == "loiter":
            a[:3] = toward(hand, [goal[0], goal[1], goal[2]], 10.0) + self.rng.normal(0, 0.4, 3)
            a[3] = close
            return np.clip(a, -1.0, 1.0)
        if self.phase == 0:
            a[:3], a[3] = toward(hand, [grasp_xy[0], grasp_xy[1], hover_z], 10.0), opened
            if np.linalg.norm(hand[:2] - grasp_xy) < 0.018 and abs(hand[2] - hover_z) < 0.035 or self.phase_steps > 60:
                self._advance()
        elif self.phase == 1:
            target = np.array([grasp_xy[0], grasp_xy[1], grasp_z])
            a[:3], a[3] = toward(hand, target, 10.0), opened
            if np.linalg.norm(hand - target) < 0.025 or self.phase_steps > 45:
                self._advance()
        elif self.phase == 2:
            a[:3], a[3] = toward(hand, [grasp_xy[0], grasp_xy[1], grasp_z], 5.0), close
            if self.phase_steps > 40:
                self._advance()
        elif self.phase == 3:
            a[:3], a[3] = toward(hand, [grasp_xy[0], grasp_xy[1], lift_z], 10.0), close
            if obj[2] > max(0.08, goal[2] - 0.04) or hand[2] > lift_z - 0.025 or self.phase_steps > 70:
                self._advance()
        elif self.phase == 4:
            a[:3], a[3] = toward(hand, [goal[0], goal[1], goal_hover_z], 10.0), close
            if np.linalg.norm(hand[:2] - goal[:2]) < 0.03 or self.phase_steps > 90:
                self._advance()
        elif self.phase == 5:
            a[:3], a[3] = toward(hand, [goal[0], goal[1], goal[2]], 10.0), close
            if np.linalg.norm(obj - goal) < 0.07 or self.phase_steps > 80:
                self._advance()
        else:
            a[:3], a[3] = toward(hand, [goal[0], goal[1], goal[2] + 0.03], 6.0), opened
        self.phase_steps += 1
        return np.clip(a, -1.0, 1.0)

    def _advance(self):
        self.phase += 1
        self.phase_steps = 0


def state(obs, goal):
    return {"hand": np.asarray(obs[:3], dtype=np.float64), "gripper": float(obs[3]),
            "obj": np.asarray(obs[4:7], dtype=np.float64), "goal": goal}


def score(s):
    d_hand_obj = np.linalg.norm(s["hand"] - s["obj"])
    d_obj_goal = np.linalg.norm(s["obj"] - s["goal"])
    reach = 1.0 - np.clip((d_hand_obj - 0.04) / 0.11, 0.0, 1.0)
    lift = np.clip(max(0.0, s["obj"][2] - 0.02) / 0.10, 0.0, 1.0)
    grasp = max(lift, reach * np.clip((1.0 - s["gripper"]) / 0.5, 0.0, 1.0))
    place = lift * (1.0 - np.clip((d_obj_goal - 0.04) / 0.11, 0.0, 1.0))
    return {"reach": float(reach), "grasp": float(grasp), "lift": float(lift), "place": float(place)}


def make_env(seed):
    bench = metaworld.MT1(TASK, seed=seed)
    env = bench.train_classes[TASK]()
    tasks = [t for t in bench.train_tasks if t.env_name == TASK]
    env.set_task(tasks[seed % len(tasks)])
    env.action_space.seed(seed)
    return env


def run_session(seed, close_sign, stall_spans=(), injection=None, random_policy=False):
    env = make_env(seed)
    obs, _ = env.reset(seed=seed)
    goal = np.asarray(env._target_pos, dtype=np.float64)
    policy = PickPlace(close_sign, np.random.default_rng(seed))
    monitor = Monitor(SCHEMA)
    records, success = [], False
    for t in range(1, MAX_EXCHANGES + 1):
        start = state(obs, goal)
        stall = any(a <= t < b for a, b in stall_spans)
        window, success_now, done = [], False, False
        for _ in range(K_STEPS):
            action = env.action_space.sample() if random_policy else policy.act(obs, goal, injection if stall else None)
            obs, _, term, trunc, info = env.step(action)
            window.append(state(obs, goal))
            success_now = success_now or bool(info.get("success", False))
            done = term or trunc
            if done:
                break
        success = success or success_now
        end = window[-1]
        embedding = np.concatenate([np.mean([w["hand"] for w in window], axis=0),
                                    np.mean([w["obj"] for w in window], axis=0),
                                    [np.mean([w["gripper"] for w in window])],
                                    end["hand"] - start["hand"], end["obj"] - start["obj"]])
        scores = score(end)
        delta = {d: scores[d] - monitor.upsilon[d] for d in SCHEMA}
        target = max(delta, key=delta.get)
        if delta[target] <= 0:
            target = max(SCHEMA, key=lambda d: scores[d] * (1.0 - monitor.upsilon[d]))
        pe, si, gains = monitor.step(embedding, scores, target)
        records.append({"t": t, "target": target, "si": si, "pe": pe, "gain": sum(gains.values()),
                        "upsilon": dict(monitor.upsilon), "stall_gt": stall, "engine_success": success_now})
        if done:
            break
    env.close()
    return records, success


def qualify(seeds):
    working = {}
    for seed in seeds:
        for sign in (1.0, -1.0):
            if run_session(seed, sign)[1]:
                working[seed] = sign
                break
    return working
