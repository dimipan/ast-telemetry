import argparse

import numpy as np

from asttel import dialogue, trajectory
from asttel.data import load_embeddings, load_json, output_dir, write_json
from asttel.stats import argmax, binomial_greater, sign_test


def target_of(scores):
    if not scores:
        return "general"
    best = argmax(scores)
    return best if scores[best] > 0 else "general"


def to_turns(session):
    landmarks = session["shared_landmarks"]
    turns = []
    for qa in session["qa_pairs"]:
        scores = {lm: float(qa["knowledge_scores"].get(lm, 0.0)) for lm in landmarks}
        turns.append({"question": qa["question"], "answer": qa["answer"],
                      "target_dimension": target_of(scores), "knowledge_values": scores})
    return turns


def next_gain_prediction(session, records, k):
    qa = session["qa_pairs"]
    n_landmarks = len(session["shared_landmarks"])
    hits = recency = uniform = n = 0
    for t in range(len(qa) - 1):
        mentioned = {lm for tt in range(t + 1, min(len(qa), t + 1 + k))
                     for lm, v in qa[tt]["knowledge_scores"].items() if v > 0.0}
        if not mentioned:
            continue
        recent, best = None, 0.0
        for lm, v in qa[t]["knowledge_scores"].items():
            if v > best:
                recent, best = lm, v
        n += 1
        hits += argmax(records[t]["pe"]) in mentioned
        uniform += len(mentioned) / n_landmarks
        recency += recent is not None and recent in mentioned
    return {"n": n, "pe": hits, "uniform": uniform, "recency": recency}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=5)
    args = p.parse_args()

    out = output_dir("study2_maptask")
    embeddings = load_embeddings("maptask")
    sessions = load_json("maptask", "annotations.json")["sessions"]
    rows = []
    for s in sessions:
        landmarks = s["shared_landmarks"]
        records = dialogue.run(to_turns(s), landmarks, embeddings)
        trajectory.write(out / "trajectories" / f"{s['dialogue_id']}.jsonl",
                         {"study": "maptask", "session": s["dialogue_id"], "schema": landmarks}, records)
        rows.append({"session": s["dialogue_id"], **dialogue.summary(records, landmarks),
                     **next_gain_prediction(s, records, args.k)})

    n = sum(r["n"] for r in rows)
    pe, uniform, recency = (sum(r[key] for r in rows) / n for key in ("pe", "uniform", "recency"))
    eligible = [r for r in rows if r["n"] > 0]
    above = sum(r["pe"] / r["n"] > r["uniform"] / r["n"] for r in eligible)
    below = sum(r["pe"] / r["n"] < r["uniform"] / r["n"] for r in eligible)
    result = {
        "k": args.k,
        "n_sessions": len(rows),
        "n_eval": n,
        "pe_top1": pe,
        "uniform": uniform,
        "recency": recency,
        "pe_minus_uniform": pe - uniform,
        "binomial_p": binomial_greater(int(sum(r["pe"] for r in rows)), n, uniform),
        "sessions_pe_above_uniform": f"{above}/{len(eligible)}",
        "sign_test_p": sign_test(above, below),
        "mean_stall_burden": float(np.mean([r["stall_burden"] for r in rows])),
    }
    write_json(out / "summary.json", {"result": result, "sessions": rows})
    for key, value in result.items():
        print(f"{key:28s} {value}")


if __name__ == "__main__":
    main()
