import argparse
import json
from pathlib import Path

REFERENCE = Path(__file__).resolve().parent.parent / "reference"
TOLERANCE = {"study3_metaworld": 5e-2, "study4": 1e-3}


def flatten(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flatten(v, f"{prefix}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from flatten(v, f"{prefix}[{i}]")
    else:
        yield prefix, obj


def compare(reference, result):
    ref, res = dict(flatten(reference)), dict(flatten(result))
    worst, mismatched = 0.0, []
    for key, expected in ref.items():
        got = res.get(key)
        if isinstance(expected, (int, float)) and isinstance(got, (int, float)):
            worst = max(worst, abs(expected - got))
        elif expected != got:
            mismatched.append(key)
    return worst, mismatched


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outputs", default="outputs")
    args = p.parse_args()
    for ref in sorted(REFERENCE.glob("*.json")):
        study = ref.stem
        name = "table1.json" if study == "study4" else "summary.json"
        result = Path(args.outputs) / study / name
        if not result.exists():
            print(f"{study:22s} not run")
            continue
        worst, mismatched = compare(json.loads(ref.read_text()), json.loads(result.read_text()))
        status = "ok" if worst <= TOLERANCE.get(study, 1e-6) and not mismatched else "differs"
        print(f"{study:22s} {status:8s} max |diff| {worst:.2e}" + (f"  mismatched {mismatched[:3]}" if mismatched else ""))


if __name__ == "__main__":
    main()
