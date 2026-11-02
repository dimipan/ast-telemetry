import numpy as np

from asttel.data import load_json, output_dir, write_json

CORPORA = ["hiker", "elderly", "child"]


def krippendorff_interval(ratings):
    units = [u[~np.isnan(u)] for u in ratings.T]
    units = [u for u in units if len(u) > 1]
    observed = sum(((u[:, None] - u[None, :]) ** 2).sum() / (len(u) - 1) for u in units)
    values = np.concatenate(units)
    expected = ((values[:, None] - values[None, :]) ** 2).sum() / (len(values) - 1)
    return float(1.0 - observed / expected)


def matrix(corpus):
    data = load_json("sar", "llm_scores", f"{corpus}.json")
    dims, models = data["dimensions"], data["models"]
    llm = np.array([[t["model_scores"][m][d] for t in data["per_turn"] for d in dims] for m in models])
    manual = np.array([t["manual_scores"][d] for t in data["per_turn"] for d in dims])
    return llm, manual


def summary(llm, manual):
    ensemble = llm.mean(axis=0)
    return {"units": int(llm.shape[1]), "alpha_llm": krippendorff_interval(llm),
            "alpha_llm_and_expert": krippendorff_interval(np.vstack([llm, manual])),
            "mae_ensemble_vs_expert": float(np.mean(np.abs(ensemble - manual))),
            "pearson_ensemble_vs_expert": float(np.corrcoef(ensemble, manual)[0, 1])}


def main():
    out = output_dir("study1_agreement")
    parts = {c: matrix(c) for c in CORPORA}
    result = {c: summary(*parts[c]) for c in CORPORA}
    result["pooled"] = summary(np.hstack([parts[c][0] for c in CORPORA]), np.concatenate([parts[c][1] for c in CORPORA]))
    write_json(out / "summary.json", result)
    for k, v in result.items():
        print(f"{k:8s}", "  ".join(f"{m} {x:.4f}" if isinstance(x, float) else f"{m} {x}" for m, x in v.items()))


if __name__ == "__main__":
    main()
