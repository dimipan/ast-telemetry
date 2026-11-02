import argparse
import json
import re

import numpy as np
import requests

from asttel.data import data_root, load_json

URL = "http://localhost:11434/api/chat"
MODELS = ["gemma3:12b", "ministral-3:14b", "gemma2", "qwen3", "llama3.1"]
DOMAIN = {
    "hiker": "Search and rescue interview about a missing adult hiker on Pine Ridge trail.",
    "elderly": "Search and rescue interview about a missing elderly person with dementia (Margaret Chen, 78).",
    "child": "Search and rescue interview about a missing child who has not returned from school (Emma Rodriguez, 9).",
}
SAR_SYSTEM_PROMPT = """You are an expert scorer of information content in search-and-rescue interviews.

Your task is to score how much operationally useful information a single question-answer exchange CONTAINS for each SAR knowledge dimension.

Score ONLY what the ANSWER says — not the quality of the question.
Score each answer IN ISOLATION — treat it as if it is the only thing you have been told.

THE DIMENSIONS

Each corpus uses a specific set of dimensions (provided in the user message).
Score ONLY those dimensions. Return 0.0 for any dimension the answer does not address.

WHAT A SCORE MEANS

A score represents the COMPLETENESS of coverage for a dimension:
  1.0  = this answer alone fully covers everything a search team could need for this dimension
  0.5  = this answer provides partial, useful but incomplete coverage
  0.0  = this answer contains nothing relevant for this dimension

Do NOT score relative to what previous turns said. Do NOT score marginal gains.
Score the absolute information content of this answer, read on its own.

SCORING RUBRIC

0.00  No relevant information.
  The answer is genuinely empty for this dimension.
  Examples: "Yeah.", "Maybe.", "I don't know.", "I've already told you.", pure repetition.

0.10–0.25  Weak but real content.
  Something relevant is present but vague, brief, or only a faint hint.
  Examples:
    "He left sometime in the afternoon." → time 0.15
    "He was wearing something red." → description 0.12
    "She might have gone toward the town." → location 0.18

0.35–0.60  Moderate, useful content.
  Conveys something a search team would act on — approximate time, general area,
  recognisable clothing, a named person, a likely destination.
  Examples:
    "He texted around 2 PM when he arrived." → time 0.45
    "He was wearing a red jacket and tan trousers." → description 0.50
    "He's diabetic and needs insulin daily." → medical 0.55

0.70–0.95  Strong, specific content.
  Concrete, detailed, operationally actionable. Named places, exact times,
  route structure, distinctive features, explicit plans, critical medical details.
  Examples:
    "He texted at 2:17 PM and sent a trail-map photo at 2:22 PM." → time 0.88
    "Pine Ridge trailhead, eastern forest entrance off Highway 16, 5 miles past the ranger station." → location 0.85
    "Insulin-dependent diabetic using a pen; misses a dose within 2-3 hours if no food." → medical 0.90

INTERMEDIATE VALUES

Use any value that fits the evidence:
  0.05 = borderline zero — technically non-empty but barely above nothing
  0.28 = slightly stronger than weak but not yet moderate
  0.65 = between moderate and strong
Do not default to round numbers like 0.0, 0.5, or 1.0. Match the number to the actual content.

CALIBRATION REMINDERS

- Exact timestamps, phone logs, CCTV times, medication schedules → raise time and medical scores
- Named places + route structure, forks, entrances, landmarks → raise location and route scores
- Distinctive clothing, visible identifiers, scars, accessories → raise description scores
- Named contacts with phone numbers → raise companions/contacts scores
- Emotional intensity alone does NOT raise a score unless it reveals substantive content
- Short acknowledgements ("Yes", "Okay", "I agree") score 0.0 on ALL dimensions

DO NOT
- infer facts not explicitly stated in the answer
- reward the question's quality or phrasing
- give non-zero scores because the answer is on-topic but empty
- output reasoning, preamble, or markdown — JSON only

The dimensions to score and the exact JSON format to return are specified in each user message."""


def user_message(domain, question, answer, dims):
    fields = "\n".join(f'  "{d}": 0.0' for d in dims)
    return (f"DOMAIN: {domain}\n\nQUESTION: {question}\n\nANSWER: {answer}\n\n"
            "Task: Score the answer's information gain for each SAR knowledge dimension.\n"
            "Important: score only what this answer explicitly adds. "
            "If context is provided above, score only the incremental gain beyond it.\n\n"
            f"Return ONLY valid JSON in exactly this structure:\n{{\n{fields}\n}}")


def parse(text, dims):
    text = re.sub(r"```(?:json)?", "", text)
    raw = text[text.find("{"):text.rfind("}") + 1]
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        obj = json.loads(re.sub(r",\s*([}\]])", r"\1", re.sub(r"(//|#)[^\n]*", "", raw)))
    value = lambda v: float(v.get("score", 0.0)) if isinstance(v, dict) else float(v)
    return {d: round(min(max(value(obj.get(d, 0.0)), 0.0), 1.0), 4) for d in dims}


def score(model, domain, question, answer, dims, url):
    r = requests.post(url, timeout=180, json={
        "model": model, "stream": False, "options": {"temperature": 0.0},
        "messages": [{"role": "system", "content": SAR_SYSTEM_PROMPT},
                     {"role": "user", "content": user_message(domain, question, answer, dims)}]})
    r.raise_for_status()
    try:
        return parse(r.json()["message"]["content"], dims)
    except (ValueError, json.JSONDecodeError):
        return {d: 0.0 for d in dims}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("corpora", nargs="*", default=list(DOMAIN))
    p.add_argument("--url", default=URL)
    args = p.parse_args()
    for name in args.corpora:
        corpus = load_json("sar", f"{name}.json")
        dims, rows = corpus["dimensions"], []
        for scenario, turns in corpus["scenarios"].items():
            for t, turn in enumerate(turns, start=1):
                if not turn.get("knowledge_values"):
                    continue
                models = {m: score(m, DOMAIN[name], turn["question"], turn["answer"], dims, args.url) for m in MODELS}
                rows.append({"scenario_id": scenario, "turn_number": t, "question": turn["question"],
                             "answer": turn["answer"],
                             "manual_scores": {d: float(turn["knowledge_values"].get(d, 0.0)) for d in dims},
                             "model_scores": models,
                             "ensemble_mean": {d: float(np.mean([models[m][d] for m in MODELS])) for d in dims}})
        path = data_root() / "sar" / "llm_scores" / f"{name}.json"
        path.write_text(json.dumps({"dimensions": dims, "models": MODELS, "per_turn": rows}, indent=1))
        print(f"{name}: {len(rows)} turns -> {path}")


if __name__ == "__main__":
    main()
