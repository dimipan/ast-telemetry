import argparse

from asttel import dialogue, trajectory
from asttel.data import load_embeddings, load_json, output_dir, write_json

CORPORA = ["hiker", "elderly", "child"]
REGIMES = {"S1_efficient": "S1", "S2_agent": "S2", "S3_interviewee": "S3"}
KEYS = ["stall_burden", "mean_si", "total_knowledge", "categories_resolved"]


def ensemble_scores(corpus):
    rows = load_json("sar", "llm_scores", f"{corpus}.json")["per_turn"]
    return {(r["scenario_id"], int(r["turn_number"])): r["ensemble_mean"] for r in rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", choices=["expert", "ensemble"], default="expert")
    args = p.parse_args()

    out = output_dir(f"study1_sar_{args.scores}")
    embeddings = load_embeddings("sar")
    runs = []
    for corpus in CORPORA:
        c = load_json("sar", f"{corpus}.json")
        lookup = ensemble_scores(corpus) if args.scores == "ensemble" else {}
        for scenario, regime in REGIMES.items():
            turns = [dict(turn) for turn in c["scenarios"][scenario]]
            for t, turn in enumerate(turns, start=1):
                if (scenario, t) in lookup:
                    turn["knowledge_values"] = lookup[(scenario, t)]
            records = dialogue.run(turns, c["dimensions"], embeddings, c["importance"], c["dependencies"])
            trajectory.write(out / "trajectories" / f"{corpus}_{regime}.jsonl",
                             {"study": "sar", "corpus": corpus, "regime": regime, "schema": c["dimensions"]}, records)
            runs.append({"corpus": corpus, "regime": regime, **dialogue.summary(records, c["dimensions"])})

    table = {r: dialogue.mean_sd([x for x in runs if x["regime"] == r], KEYS) for r in REGIMES.values()}
    write_json(out / "summary.json", {"scores": args.scores, "runs": runs, "by_regime": table})
    for regime, row in table.items():
        print(regime, "  ".join(f"{k} {v['mean']:.2f} ± {v['sd']:.2f}" for k, v in row.items()))


if __name__ == "__main__":
    main()
