import argparse
import itertools

import numpy as np
import requests

from asttel.data import data_root, load_json
from asttel.rl.env import InterviewEnv, format_question

MODEL = "qwen3-embedding"
URL = "http://localhost:11434/api/embeddings"
SAR = ["hiker", "elderly", "child"]


def sar_texts():
    texts = set()
    for name in SAR:
        corpus = load_json("sar", f"{name}.json")
        texts |= set(corpus["answers"].values()) | set(corpus["answer_scores"])
        texts |= {turn["answer"] for turns in corpus["scenarios"].values() for turn in turns}
        env = InterviewEnv(corpus, {}, "Baseline", "A")
        for mask in itertools.product([0.0, 1.0], repeat=len(env.dims)):
            upsilon = dict(zip(env.dims, mask))
            for strategy, dim in env.actions:
                template = env.questions.get((strategy, dim)) or env.questions.get((strategy, "general"))
                question = format_question(template, upsilon)
                texts.add(corpus["answers"].get(question, f"I'm not sure about {dim} related to '{question[:30]}...'."))
    return texts


def session_texts(*parts):
    return {qa["answer"] for s in load_json(*parts)["sessions"] for qa in s["qa_pairs"]}


def annomi_texts():
    data = load_json("annomi", "annotations.json")
    return {qa["answer"] for key, s in data.items() if key != "_meta" for qa in s["qa_pairs"]}


TEXTS = {"sar": sar_texts, "maptask": lambda: session_texts("maptask", "annotations.json"),
         "multiwoz": lambda: session_texts("multiwoz", "sessions.json"), "annomi": annomi_texts}


def embed(text, url, model):
    r = requests.post(url, json={"model": model, "prompt": text}, timeout=120)
    r.raise_for_status()
    return np.asarray(r.json()["embedding"], dtype=np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("names", nargs="+", choices=list(TEXTS))
    p.add_argument("--url", default=URL)
    p.add_argument("--model", default=MODEL)
    args = p.parse_args()
    for name in args.names:
        texts = sorted(TEXTS[name]())
        vectors = np.stack([embed(t, args.url, args.model) for t in texts])
        path = data_root() / "embeddings" / f"{name}.npz"
        np.savez_compressed(path, texts=np.array(texts), vectors=vectors)
        print(f"{name}: {len(texts)} texts, dim {vectors.shape[1]} -> {path}")


if __name__ == "__main__":
    main()
