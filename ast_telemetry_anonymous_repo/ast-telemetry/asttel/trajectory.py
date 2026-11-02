import json
from pathlib import Path


def record(t, target, monitor, pe, si, gains, **extra):
    return {
        "t": t,
        "target": target,
        "upsilon": dict(monitor.upsilon),
        "gain": {d: g for d, g in gains.items() if g > 0},
        "pe": pe,
        "si": si,
        **extra,
    }


def write(path, header, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(json.dumps({"header": header}) + "\n")
        for r in records:
            f.write(json.dumps(r) + "\n")


def read(path):
    lines = Path(path).read_text().splitlines()
    return json.loads(lines[0])["header"], [json.loads(x) for x in lines[1:]]
