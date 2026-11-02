import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

SLOTS = ["departure", "destination", "day", "leaveat", "arriveby", "people"]
ALIASES = {"leave at": "leaveat", "leave": "leaveat", "arrive by": "arriveby", "arrive": "arriveby",
           "depart": "departure", "dest": "destination"}
EMPTY = {"", "?", "dontcare", "not mentioned", "none", "null"}
OTHER_DOMAINS = ["hotel", "restaurant", "attraction", "hospital", "police"]


def canonical(raw):
    s = raw.strip().lower().replace(" ", "")
    if s in SLOTS:
        return s
    if raw.strip().lower() in ALIASES:
        return ALIASES[raw.strip().lower()]
    return next((d for d in SLOTS if s.startswith(d) or d.startswith(s)), None)


def filled(value):
    return bool(value) and value.strip().lower() not in EMPTY


def belief_scores(metadata):
    train = metadata.get("train", {})
    state = {}
    for raw, value in train.get("semi", {}).items():
        slot = canonical(raw)
        if slot:
            state[slot] = value
    people = train.get("book", {}).get("people")
    if people is not None and filled(str(people)):
        state["people"] = str(people)
    return {d: 1.0 if filled(state.get(d, "")) else 0.0 for d in SLOTS}


def is_train(dialogue):
    goal = dialogue.get("goal", {})
    if not goal.get("train"):
        return False
    return not any(goal.get(d, {}).get("info") or goal.get(d, {}).get("reqt") for d in OTHER_DOMAINS)


def booked(dialogue):
    return any(t.get("metadata", {}).get("train", {}).get("book", {}).get("booked") for t in dialogue.get("log", []))


def parse(dialogue_id, dialogue):
    log = dialogue.get("log", [])
    pairs, i = [], 1
    while i + 1 < len(log):
        source = log[i + 2] if i + 2 < len(log) else log[i]
        pairs.append({"turn_number": len(pairs) + 1, "question": log[i].get("text", "").strip(),
                      "answer": log[i + 1].get("text", "").strip(),
                      "knowledge_scores": belief_scores(source.get("metadata", {}))})
        i += 2
    if len(pairs) < 2:
        return None
    peak, prev = {d: 0.0 for d in SLOTS}, {d: 0.0 for d in SLOTS}
    for qa in pairs:
        for d in SLOTS:
            peak[d] = max(qa["knowledge_scores"][d], peak[d])
            qa["knowledge_scores"][d] = peak[d]
        qa["belief_delta"] = {d: max(0.0, peak[d] - prev[d]) for d in SLOTS}
        prev = dict(peak)
    return {"dialogue_id": dialogue_id, "group": "success" if booked(dialogue) else "failure", "qa_pairs": pairs}


def sample(sessions, n, seed):
    rng = random.Random(seed)
    cells = defaultdict(list)
    for s in sessions:
        k = len(s["qa_pairs"])
        cells[(s["group"], "short" if k <= 6 else "long" if k >= 9 else "medium")].append(s)
    for pool in cells.values():
        rng.shuffle(pool)
    targets = {("success", "short"): n // 4, ("success", "long"): n // 4, ("failure", "short"): n // 5,
               ("failure", "long"): n // 5, ("success", "medium"): n // 10, ("failure", "medium"): n // 10}
    selected = [s for cell, k in targets.items() for s in cells.get(cell, [])[:max(1, k)]]
    seen = {s["dialogue_id"] for s in selected}
    for s in sessions:
        if len(selected) >= n:
            break
        if s["dialogue_id"] not in seen:
            selected.append(s)
            seen.add(s["dialogue_id"])
    rng.shuffle(selected)
    return selected[:n]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("multiwoz_data_json")
    p.add_argument("output")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    raw = json.loads(Path(args.multiwoz_data_json).read_text())
    sessions = [s for did, d in sorted(raw.items()) if is_train(d) for s in [parse(did, d)] if s]
    chosen = sample(sessions, args.n, args.seed)
    Path(args.output).write_text(json.dumps({"schema": SLOTS, "sessions": chosen}, indent=1))
    print(f"{len(chosen)} sessions written to {args.output}")


if __name__ == "__main__":
    main()
