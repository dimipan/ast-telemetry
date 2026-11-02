import re

import numpy as np

from asttel import dialogue, trajectory
from asttel.data import load_embeddings, load_json, output_dir, write_json
from asttel.stats import argmax, binomial_greater

PATTERNS = [
    ("departure", [r"depart(?:ing|ure)? from", r"leaving from", r"leave from", r"coming from",
                   r"travel(?:ling)? from", r"where.*(?:are|do) you.*(?:leave|depart)"]),
    ("destination", [r"destination", r"going to", r"travel(?:ling)? to", r"headed to", r"heading to",
                     r"where.*would you like to go", r"where.*(?:are|do) you.*go", r"arrive at", r"arriving at"]),
    ("day", [r"what day", r"which day", r"what date", r"when would you", r"day of", r"travel(?:ling)? on",
             r"leaving on"]),
    ("leaveat", [r"leave at", r"leaving at", r"depart at", r"time.*(?:would|to) leave", r"what time.*leave",
                 r"after \d{2}:\d{2}"]),
    ("arriveby", [r"arrive by", r"arrival time", r"arriving by", r"time.*arrive", r"need to arrive",
                  r"what time.*arrive", r"before \d{2}:\d{2}"]),
    ("people", [r"how many (?:people|passengers|tickets|persons)", r"number of (?:people|passengers|tickets)",
                r"party size", r"travelling alone", r"traveling alone", r"for (?:how many|one|two|three|\d+)"]),
]


def target_of(question, delta):
    q = question.lower()
    for slot, patterns in PATTERNS:
        if any(re.search(p, q) for p in patterns):
            return slot
    gained = [d for d, v in delta.items() if v > 0]
    return max(delta, key=delta.get) if gained else None


def next_slot_prediction(session, records):
    qa = session["qa_pairs"]
    pe = uniform = recency = n = 0
    for t in range(len(qa) - 1):
        nxt = {d for d, v in qa[t + 1]["belief_delta"].items() if v > 0}
        if not nxt:
            continue
        recent = next((argmax(qa[s]["belief_delta"]) for s in range(t, -1, -1)
                       if any(v > 0 for v in qa[s]["belief_delta"].values())), None)
        n += 1
        pe += argmax(records[t]["pe"]) in nxt
        uniform += len(nxt) / len(records[t]["pe"])
        recency += recent in nxt
    return {"n": n, "pe": pe, "uniform": uniform, "recency": recency}


def main():
    out = output_dir("study2_multiwoz")
    embeddings = load_embeddings("multiwoz")
    data = load_json("multiwoz", "sessions.json")
    schema, rows = data["schema"], []
    for s in data["sessions"]:
        turns = [{"question": qa["question"], "answer": qa["answer"], "knowledge_values": qa["knowledge_scores"],
                  "target_dimension": target_of(qa["question"], qa["belief_delta"])} for qa in s["qa_pairs"]]
        records = dialogue.run(turns, schema, embeddings, {d: 1.0 for d in schema})
        trajectory.write(out / "trajectories" / f"{s['dialogue_id'].removesuffix('.json')}.jsonl",
                         {"study": "multiwoz", "session": s["dialogue_id"], "schema": schema}, records)
        rows.append({"session": s["dialogue_id"], "group": s["group"],
                     **dialogue.summary(records, schema), **next_slot_prediction(s, records)})

    n = sum(r["n"] for r in rows)
    hits = sum(r["pe"] for r in rows)
    uniform = sum(r["uniform"] for r in rows) / n
    result = {
        "n_sessions": len(rows),
        "n_eval": n,
        "pe_top1": hits / n,
        "uniform": uniform,
        "recency": sum(r["recency"] for r in rows) / n,
        "binomial_p_vs_uniform": binomial_greater(hits, n, uniform),
        "binomial_p_vs_one_over_n": binomial_greater(hits, n, 1 / len(schema)),
        "fraction_sessions_zero_stall": float(np.mean([r["stall_burden"] == 0 for r in rows])),
    }
    write_json(out / "summary.json", {"result": result, "sessions": rows})
    for key, value in result.items():
        print(f"{key:30s} {value}")


if __name__ == "__main__":
    main()
