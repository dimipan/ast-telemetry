import json
import os
from pathlib import Path

import numpy as np


def data_root():
    root = os.environ.get("AST_DATA")
    if not root:
        raise SystemExit("Set AST_DATA to the path of the AST data directory.")
    path = Path(root).expanduser()
    if not path.is_dir():
        raise SystemExit(f"AST_DATA does not exist: {path}")
    return path


def data_path(*parts):
    path = data_root().joinpath(*parts)
    if not path.exists():
        raise SystemExit(f"Missing data file: {path}")
    return path


def load_json(*parts):
    return json.loads(data_path(*parts).read_text())


def load_embeddings(name):
    z = np.load(data_path("embeddings", f"{name}.npz"))
    return dict(zip(z["texts"].tolist(), z["vectors"]))


def output_dir(study):
    path = Path(os.environ.get("AST_OUTPUT", "outputs")) / study
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, default=float))
