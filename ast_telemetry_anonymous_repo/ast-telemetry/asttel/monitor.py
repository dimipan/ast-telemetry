import math

import numpy as np

from asttel.params import THETA


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x)
    return x / n if n > 1e-12 else x


def residual(V, r, eps):
    c = np.linalg.lstsq(V, r, rcond=None)[0]
    return min(max(float(np.linalg.norm(r - V @ c) / (np.linalg.norm(r) + eps)), 0.0), 1.0)


class Monitor:
    def __init__(self, schema, importance=None, dependencies=None, aggregate_gain=False, theta=THETA):
        self.schema = list(schema)
        self.importance = dict(importance or {})
        self.dependencies = dict(dependencies or {})
        self.aggregate_gain = aggregate_gain
        self.theta = theta
        self.upsilon = {d: 0.0 for d in self.schema}
        self.attempts = {d: 0 for d in self.schema}
        self.successes = {d: 0 for d in self.schema}
        self.history = {d: [] for d in self.schema}
        self.track = {}
        self.checkpoints = {d: [] for d in self.schema}

    def step(self, embedding, scores, target):
        if scores is None:
            return self.progress(), 0.0, {d: 0.0 for d in self.schema}
        th = self.theta
        r = unit(embedding)
        if not self.track:
            self.track = {d: np.zeros(r.shape[0]) for d in self.schema}
        old = dict(self.upsilon)
        for d, v in scores.items():
            if d in self.upsilon and v > self.upsilon[d]:
                self.upsilon[d] = float(v)
        gains = {d: max(0.0, scores.get(d, 0.0) - old[d]) for d in self.schema}
        attributed = target in self.upsilon
        if attributed and not self.aggregate_gain:
            gain = scores.get(target, 0.0) - old[target]
        else:
            gain = sum(gains.values())
        si = self.friction(r, gain)
        if attributed:
            self.attempts[target] += 1
            self.successes[target] += scores.get(target, 0.0) - old[target] > th.tau_g
            self.history[target].append(r)
        for d, v in scores.items():
            if d in self.upsilon and v > 0 and v > old[d]:
                self.checkpoints[d].append(self.track[d].copy())
                self.track[d] = self.track[d] + r * (v - old[d])
        return self.progress(), si, gains

    def friction(self, r, gain):
        th = self.theta
        if np.linalg.norm(r) < th.eps_g:
            return 0.0
        damp = max(th.d_min, 1.0 - th.lam * gain)
        num = den = 0.0
        for d in self.schema:
            if not self.history[d]:
                continue
            mu = 1.0 - residual(np.column_stack(self.history[d]), r, th.eps_g)
            num += mu * mu * damp * (1.0 - self.upsilon[d])
            den += mu
        if den < th.eps_g:
            return 0.0
        return float(max(0.0, min(1.0, num / (den + th.eps_g))))

    def progress(self):
        th = self.theta
        beta = {d: (1.0 + self.successes[d], 1.0 + self.attempts[d] - self.successes[d]) for d in self.schema}
        var = {d: a * b / ((a + b) ** 2 * (a + b + 1.0)) for d, (a, b) in beta.items()}
        max_var = max([th.eps_p, *var.values()])
        pe = {}
        for d in self.schema:
            a, b = beta[d]
            v = min(max(self.upsilon[d], th.eps_p), 1.0 - th.eps_p)
            h = -(v * math.log2(v) + (1.0 - v) * math.log2(1.0 - v))
            eig = a / (a + b) * h * (1.0 + th.beta * var[d] / max_var)
            value = (th.alpha * eig + (1.0 - th.alpha) * self.trajectory(d)) * self.importance.get(d, 1.0)
            for j in self.dependencies.get(d, []):
                aj, bj = 1.0 + self.successes.get(j, 0), 1.0 + self.attempts.get(j, 0) - self.successes.get(j, 0)
                value *= aj / (aj + bj)
            pe[d] = float(value)
        return pe

    def trajectory(self, d):
        th = self.theta
        if not self.checkpoints[d]:
            return 1.0
        recent = (self.checkpoints[d] + [self.track[d]])[-th.n:]
        length = sum(np.linalg.norm(recent[j] - recent[j - 1]) for j in range(1, len(recent)))
        straightness = np.linalg.norm(recent[-1] - recent[0]) / (length + th.eps_g)
        return 1.0 - straightness * min(1.0, self.upsilon[d] / th.upsilon_star)
