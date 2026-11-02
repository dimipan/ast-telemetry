import numpy as np

from asttel import trajectory
from asttel.monitor import Monitor
from asttel.params import THETA_SI, UPSILON_COMPLETE


def run(turns, schema, embeddings, importance=None, dependencies=None):
    monitor = Monitor(schema, importance, dependencies)
    records = []
    for t, turn in enumerate(turns, start=1):
        target = turn.get("target_dimension") or infer_target(turn["question"], schema)
        scores = turn.get("knowledge_values") or None
        embedding = embeddings[turn["answer"]] if scores is not None else None
        pe, si, gains = monitor.step(embedding, scores, target)
        records.append(trajectory.record(t, target, monitor, pe, si, gains))
    return records


def infer_target(question, schema):
    q = question.lower()
    return next((d for d in schema if d.lower() in q), "general")


def summary(records, schema):
    si = np.array([r["si"] for r in records])
    final = records[-1]["upsilon"]
    gains, prev = [], {d: 0.0 for d in schema}
    for r in records:
        gains.append(max(0.0, sum(r["upsilon"][d] - prev[d] for d in schema)))
        prev = r["upsilon"]
    return {
        "n_turns": len(records),
        "stall_burden": 100.0 * float(np.mean(si > THETA_SI)),
        "mean_si": float(si.mean()),
        "peak_si": float(si.max()),
        "total_knowledge": float(sum(final.values())),
        "categories_resolved": int(sum(v >= UPSILON_COMPLETE for v in final.values())),
        "mean_gain_per_turn": float(np.mean(gains)),
        "total_pe": float(sum(sum(r["pe"].values()) for r in records)),
    }


def mean_sd(rows, keys):
    return {k: {"mean": float(np.mean([r[k] for r in rows])), "sd": float(np.std([r[k] for r in rows]))} for k in keys}
