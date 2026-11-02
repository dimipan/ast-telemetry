import string

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from asttel.monitor import Monitor
from asttel.params import THETA_SI, UPSILON_COMPLETE

VARIANTS = {
    "Baseline": (("values", "turn"), None),
    "Coupled-Blind": (("values", "turn"), "operational"),
    "AST-Obs": (("values", "pe", "si", "turn"), None),
    "AST-Pen": (("values", "pe", "turn"), "friction"),
    "Full-AST": (("values", "pe", "si", "turn"), "friction"),
}
LEGACY_NAMES = {"Full-DT": "Full-AST", "DT-no-penalty": "AST-Obs", "DT-no-SI-obs": "AST-Pen"}
STEP_COST = 0.1
COMPLETION_BONUS = 10.0
KAPPA = 1.0
OPERATIONAL_UNIT = 2.0
ZERO_GAIN = 1e-3
PLACEHOLDERS = {"location": "Pine Ridge trail", "time": "2 PM", "description": "a red jacket",
                "equipment": "a backpack", "medical": "medical condition", "weather": "sunny",
                "companions": "someone", "intentions": "the summit", "destination": "the summit"}
SUMMARY_DIMS = ["location", "time", "description", "equipment", "intentions"]
SUMMARY_KNOWN = {"location": "Pine Ridge trailhead", "time": "around 2 PM yesterday", "description": "a red jacket",
                 "equipment": "a backpack with supplies", "intentions": "hike the loop trail"}
SUMMARY_UNKNOWN = {"location": "unknown location", "time": "unknown time", "description": "unknown clothing",
                   "equipment": "unknown equipment", "intentions": "unknown destination"}


def fact(dim):
    return PLACEHOLDERS.get(dim, f"{dim} details")


def format_question(template, upsilon):
    if "{" not in template:
        return template
    lower = template.lower()
    auto, indexed, named = 0, [], []
    for _, field, _, _ in string.Formatter().parse(template):
        if field is None:
            continue
        if field == "":
            auto += 1
        elif field.isdigit():
            indexed.append(int(field))
        else:
            named.append(field)
    matched = next((d for d in upsilon if d.lower() in lower), None)
    default = fact(matched) if matched else "that information"
    kwargs = {n: fact(n.lower()) if n.lower() in PLACEHOLDERS else default for n in named}
    args = [default] * max((max(indexed) + 1) if indexed else 0, auto, 1)
    if "to confirm what we know" in lower:
        values = [SUMMARY_KNOWN[d] if upsilon.get(d, 0.0) > 0.3 else SUMMARY_UNKNOWN[d] for d in SUMMARY_DIMS]
    else:
        values = [fact(d) for d, v in upsilon.items() if v > 0.3]
    args[:min(len(values), len(args))] = values[:len(args)]
    try:
        return template.format(*args, **kwargs)
    except Exception:
        try:
            return template.format(*(["that information"] * 10))
        except Exception:
            return template


class InterviewEnv(gym.Env):
    def __init__(self, corpus, embeddings, variant, condition, max_turns=50, rho=1.0):
        super().__init__()
        self.corpus, self.embeddings = corpus, embeddings
        self.dims = corpus["dimensions"]
        self.observe, self.penalty = VARIANTS[LEGACY_NAMES.get(variant, variant)]
        self.stop_on_stall = condition == "B"
        self.max_turns, self.rho = max_turns, rho
        self.questions = {(q["strategy"], q["dimension"]): q["text"] for q in corpus["questions"]}
        self.actions = [(s, d) for s in corpus["strategies"] for d in self.dims + ["general"]
                        if not (d == "general" and s in ("CLARIFICATION", "CONTEXT_REINSTATE"))
                        and ((s, d) in self.questions or (d != "general" and (s, "general") in self.questions))]
        self.action_space = spaces.Discrete(len(self.actions))
        size = sum(len(self.dims) if part in ("values", "pe") else 1 for part in self.observe)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(size,), dtype=np.float32)

    def _obs(self, pe, si):
        parts = {"values": [self.monitor.upsilon[d] for d in self.dims], "pe": [pe[d] for d in self.dims],
                 "si": [si], "turn": [self.turn / self.max_turns]}
        return np.concatenate([np.array(parts[p], dtype=np.float32) for p in self.observe])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self.monitor = Monitor(self.dims, self.corpus["importance"], self.corpus["dependencies"])
        self.turn, self.used = 0, set()
        return self._obs(self.monitor.progress(), 0.0), {}

    def step(self, action):
        self.turn += 1
        strategy, dim = self.actions[int(action)]
        template = self.questions.get((strategy, dim)) or (self.questions.get((strategy, "general")) if dim != "general" else None)
        question = format_question(template, self.monitor.upsilon)
        answer = self.corpus["answers"].get(question, f"I'm not sure about {dim} related to '{question[:30]}...'.")
        scores = self.corpus["answer_scores"].get(answer, {d: 0.01 for d in self.dims if d == dim})
        before = dict(self.monitor.upsilon)
        if scores:
            pe, si, _ = self.monitor.step(self.embeddings[answer], scores, dim)
        else:
            pe, si, _ = self.monitor.step(None, None, dim)
        gain = {d: self.monitor.upsilon[d] - before[d] for d in self.dims}
        repeat = int(action) in self.used
        self.used.add(int(action))
        cost = OPERATIONAL_UNIT * (repeat + (sum(max(0.0, g) for g in gain.values()) < ZERO_GAIN))
        reward = -STEP_COST
        if self.penalty == "friction":
            reward -= KAPPA * si
        elif self.penalty == "operational":
            reward -= self.rho * cost
        complete = sum(self.monitor.upsilon[d] >= UPSILON_COMPLETE for d in self.dims)
        terminated = complete == len(self.dims)
        if terminated:
            reward += COMPLETION_BONUS
        truncated = self.turn >= self.max_turns or (self.stop_on_stall and si > THETA_SI)
        info = {"SI": si, "PE": pe, "target": dim, "upsilon": dict(self.monitor.upsilon), "knowledge_gain": gain,
                "complete_dimensions": complete, "operational_cost": cost,
                "total_knowledge": sum(self.monitor.upsilon.values()) / len(self.dims)}
        return self._obs(pe, si), reward, terminated, truncated, info
