import numpy as np

from asttel.params import THETA_SI


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    greater = sum((p > n) for p in pos for n in neg)
    equal = sum((p == n) for p in pos for n in neg)
    return float((greater + 0.5 * equal) / (len(pos) * len(neg)))


def si_metrics(runs, warmup, close_open_spans=False):
    pos, neg, delays = [], [], []
    tp = fp = fn = 0
    for records in runs:
        in_span, fired, start = False, None, None
        for r in records:
            if r["t"] <= warmup:
                continue
            label, pred = r["stall_gt"], r["si"] > THETA_SI
            (pos if label else neg).append(r["si"])
            tp += label and pred
            fn += label and not pred
            fp += (not label) and pred
            if label and not in_span:
                in_span, start, fired = True, r["t"], None
            if label and pred and fired is None:
                fired = r["t"]
            if not label and in_span:
                in_span = False
                delays.append(fired - start if fired is not None else np.inf)
        if close_open_spans and in_span:
            delays.append(fired - start if fired is not None else np.inf)
    detected = [d for d in delays if np.isfinite(d)]
    return {
        "auc": auc(pos, neg),
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "span_coverage": len(detected) / max(1, len(delays)),
        "si_event": float(np.mean(pos)),
        "si_clean": float(np.mean(neg)),
        "median_onset_delay": float(np.median(detected)) if detected else None,
        "n_event": len(pos),
        "n_clean": len(neg),
    }


def pe_metrics(runs, warmup, horizons=(1, 3, 5), eps=1e-9):
    rng = np.random.default_rng(0)
    out = {}
    for k in horizons:
        hits = uniform = recency = n = 0
        for records in runs:
            for j in range(warmup, len(records) - k):
                base = records[j - 1]
                gained = {d for q in records[j:j + k] for d, u in q["upsilon"].items()
                          if u > base["upsilon"].get(d, 0) + eps}
                if not gained:
                    continue
                n += 1
                hits += max(base["pe"], key=base["pe"].get) in gained
                uniform += rng.choice(list(base["pe"])) in gained
                recency += base["target"] in gained
        out[f"k={k}"] = {"pe": hits / max(1, n), "recency": recency / max(1, n), "uniform": uniform / max(1, n), "n": n}
    return out


def zero_gain_fraction(runs):
    return float(np.mean([r["gain"] == 0.0 for records in runs for r in records if r["stall_gt"]]))
