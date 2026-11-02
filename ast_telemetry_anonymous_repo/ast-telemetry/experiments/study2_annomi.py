import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

from asttel import dialogue, trajectory
from asttel.data import load_embeddings, load_json, output_dir, write_json

DIMS = ["presenting_problem", "motivation_change", "barriers_concerns", "personal_resources", "action_readiness"]
IMPORTANCE = {d: 0.8 for d in DIMS}
CORRELATIONS = [
    ("stall_burden", "mean_gain_per_turn"),
    ("mean_si", "mean_gain_per_turn"),
    ("stall_burden", "total_knowledge"),
    ("mean_si", "total_knowledge"),
]


def to_turns(session):
    turns = []
    for qa in session["qa_pairs"]:
        scores = {d: float(qa["knowledge_scores"].get(d, 0.0)) for d in DIMS}
        target = max(scores, key=scores.get) if any(scores.values()) else DIMS[0]
        turns.append({"question": qa["question"], "answer": qa["answer"],
                      "target_dimension": target, "knowledge_values": scores})
    return turns


def score_gain(turns):
    gains, prev = [], {d: 0.0 for d in DIMS}
    for turn in turns:
        s = turn["knowledge_values"]
        gains.append(sum(max(0.0, s[d] - prev[d]) for d in DIMS))
        prev = s
    return float(np.mean(gains))


def main():
    out = output_dir("study2_annomi")
    embeddings = load_embeddings("annomi")
    data = load_json("annomi", "annotations.json")
    rows = []
    for sid, session in data.items():
        if sid == "_meta":
            continue
        turns = to_turns(session)
        records = dialogue.run(turns, DIMS, embeddings, IMPORTANCE)
        trajectory.write(out / "trajectories" / f"{sid}.jsonl",
                         {"study": "annomi", "session": sid, "quality": session["mi_quality"], "schema": DIMS}, records)
        row = {"session": sid, "quality": session["mi_quality"], **dialogue.summary(records, DIMS)}
        row["mean_gain_per_turn"] = score_gain(turns)
        rows.append(row)

    correlations = {}
    for x, y in CORRELATIONS:
        rho, p = spearmanr([round(r[x], 6) for r in rows], [round(r[y], 6) for r in rows])
        correlations[f"{x}_vs_{y}"] = {"rho": float(rho), "p": float(p)}
    hi = [r["stall_burden"] for r in rows if r["quality"] == "high"]
    lo = [r["stall_burden"] for r in rows if r["quality"] == "low"]
    quality = {"high_mean": float(np.mean(hi)), "low_mean": float(np.mean(lo)),
               "mann_whitney_p": float(mannwhitneyu(hi, lo, alternative="two-sided").pvalue)}
    write_json(out / "summary.json", {"n_sessions": len(rows), "correlations": correlations,
                                      "stall_burden_by_quality": quality, "sessions": rows})
    for name, c in correlations.items():
        print(f"{name:40s} rho {c['rho']:+.3f}  p {c['p']:.4f}")
    print(f"stall burden high {quality['high_mean']:.2f}  low {quality['low_mean']:.2f}  p {quality['mann_whitney_p']:.3f}")


if __name__ == "__main__":
    main()
