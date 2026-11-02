import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import json

from asttel import trajectory
from asttel.params import THETA_SI, UPSILON_COMPLETE

FLOW, STALL, PATH = "#2a9d8f", "#d1495b", "#3d405b"
LETTERS = "abcdefghijklmnopqrstuvwxyz"


def radar(ax, header, records, title):
    schema = header["schema"]
    angles = np.linspace(0, 2 * np.pi, len(schema), endpoint=False).tolist()
    ring = angles + angles[:1]
    final = [records[-1]["upsilon"][d] for d in schema]
    ax.plot(ring, [1.0] * len(ring), color="0.6", lw=1)
    ax.plot(ring, [UPSILON_COMPLETE] * len(ring), color="0.6", lw=0.8, ls="--")
    ax.fill(ring, final + final[:1], color=FLOW, alpha=0.35)
    ax.plot(ring, final + final[:1], color=FLOW, lw=1.5)
    ax.set_xticks(angles)
    ax.set_xticklabels(schema, fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels([])
    ax.set_title(title, fontsize=9, pad=16)


def phase(ax, header, records, z_max):
    schema = header["schema"]
    x = [np.mean([r["upsilon"][d] for d in schema]) for r in records]
    y = [r["si"] for r in records]
    z = [sum(r["pe"].values()) for r in records]
    ax.plot(x, y, z, color=PATH, lw=1.5)
    ax.scatter(x, y, [0] * len(z), color="0.75", s=6)
    ax.scatter(x, y, z, c=[STALL if s > THETA_SI else FLOW for s in y], s=22, depthshade=False,
               edgecolors="k", linewidths=0.3)
    ax.scatter([x[0]], [y[0]], [z[0]], marker="D", color="k", s=30)
    ax.set_xlabel("mean resolution", fontsize=7)
    ax.set_ylabel("friction $F$", fontsize=7)
    ax.set_zlabel(r"progress $\Sigma_i P_i$", fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, z_max)
    ax.tick_params(labelsize=6)


def legend(fig):
    handles = [Line2D([], [], marker="o", ls="", color=FLOW, markeredgecolor="k", label=rf"$F \leq {THETA_SI:.2f}$"),
               Line2D([], [], marker="o", ls="", color=STALL, markeredgecolor="k", label=rf"$F > {THETA_SI:.2f}$ (stall)"),
               Line2D([], [], marker="D", ls="", color="k", label="start"),
               Line2D([], [], color="0.6", ls="--", label=rf"$\upsilon_{{\mathrm{{complete}}}} = {UPSILON_COMPLETE:.2f}$")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7, frameon=False)


def figure(panels, path):
    missing = [str(f) for f, _ in panels if not Path(f).exists()]
    if missing:
        print(f"skipped {path.name}: missing {missing[0]}")
        return
    runs = [trajectory.read(f) for f, _ in panels]
    z_max = 1.05 * max(sum(r["pe"].values()) for _, records in runs for r in records)
    fig = plt.figure(figsize=(7.2, 3.2 * len(panels) + 0.4))
    for i, ((header, records), (_, title)) in enumerate(zip(runs, panels)):
        left = fig.add_subplot(len(panels), 2, 2 * i + 1, polar=True)
        right = fig.add_subplot(len(panels), 2, 2 * i + 2, projection="3d")
        radar(left, header, records, title)
        phase(right, header, records, z_max)
    legend(fig)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for i, ax in enumerate(fig.axes):
        box = ax.get_position()
        fig.text(box.x0 - 0.02, box.y1 + 0.005, LETTERS[i], fontsize=11, fontweight="bold")
    for suffix in (".png", ".pdf"):
        fig.savefig(path.with_suffix(suffix), dpi=300)
    plt.close(fig)
    print(f"wrote {path.with_suffix('.pdf')}")


def extremes(root, study, label):
    path = root / study / "summary.json"
    if not path.exists():
        return []
    rows = sorted(json.loads(path.read_text())["sessions"], key=lambda r: r["mean_si"])
    folder = root / study / "trajectories"
    return [(folder / f"{Path(r['session']).stem}.jsonl", f"{label} {Path(r['session']).stem}, {tag} mean friction")
            for r, tag in ((rows[0], "lowest"), (rows[-1], "highest"))]


def supplementary(root, out, rl_seed):
    sar = root / "study1_sar_expert" / "trajectories"
    for corpus in ("hiker", "elderly"):
        figure([(sar / f"{corpus}_{r}.jsonl", f"SAR {corpus}, regime {r}") for r in ("S1", "S2", "S3")],
               out / f"supp_sar_{corpus}.png")
    for study, label in (("study2_maptask", "Map Task"), ("study2_multiwoz", "MultiWOZ"), ("study2_annomi", "AnnoMI session")):
        panels = extremes(root, study, label)
        if panels:
            figure(panels, out / f"supp_{study.split('_')[1]}.png")
    mg = root / "study3_minigrid" / "trajectories"
    figure([(mg / "I1_s0.jsonl", "MiniGrid, loops in unexplored sectors (I1, seed 0)"),
            (mg / "I2_s0.jsonl", "MiniGrid, loops in explored sectors (I2, seed 0)")], out / "supp_minigrid.png")
    cr = root / "study3_crafter" / "trajectories"
    figure([(cr / "clean_s0.jsonl", "Crafter, clean (seed 0)"), (cr / "injected_s0.jsonl", "Crafter, injected no-op loops (seed 0)")],
           out / "supp_crafter.png")
    mw = root / "study3_metaworld" / "trajectories"
    figure([(mw / "I1_s0.jsonl", "MetaWorld, pre-grasp orbiting (I1, seed 0)"),
            (mw / "I2_s0.jsonl", "MetaWorld, post-goal holding (I2, seed 0)")], out / "supp_metaworld.png")
    bo = root / "study3_boir" / "trajectories"
    figure([(bo / "S1_01.jsonl", "BOIR, scenario S1, trial 01"), (bo / "S3_01.jsonl", "BOIR, scenario S3, trial 01")],
           out / "supp_boir.png")
    rl = root / "study4" / "trajectories"
    figure([(rl / f"{v}_condB_seed{rl_seed}_child.jsonl", f"PPO {v}, condition B") for v in ("Coupled-Blind", "AST-Obs", "AST-Pen")],
           out / "supp_rl.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outputs", default="outputs")
    p.add_argument("--rl-seed", type=int, default=0)
    p.add_argument("--trajectories", nargs="+", help="draw these trajectory files instead of the paper figures")
    p.add_argument("--name", default="custom")
    args = p.parse_args()
    root = Path(args.outputs)
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    if args.trajectories:
        figure([(Path(f), Path(f).stem) for f in args.trajectories], out / f"{args.name}.png")
        return

    sar = root / "study1_sar_expert" / "trajectories"
    figure([(sar / f"child_{r}.jsonl", f"SAR child, regime {r}") for r in ("S1", "S2", "S3")], out / "fig3_sar.png")

    mw = root / "study3_metaworld" / "trajectories"
    random_run = next(mw.glob("random_s*.jsonl"), mw / "random_s0.jsonl")
    clean_run = mw / f"C0_{random_run.stem.split('_')[1]}.jsonl"
    figure([(random_run, "MetaWorld, uniform-random policy"), (clean_run, "MetaWorld, scripted policy")], out / "fig4_metaworld.png")

    rl = root / "study4" / "trajectories"
    runs = [rl / f"{v}_condB_seed{args.rl_seed}_child.jsonl" for v in ("Baseline", "Full-AST")]
    figure([(runs[0], "PPO Baseline, condition B"), (runs[1], "PPO Full-AST, condition B")], out / "fig5_rl.png")
    supplementary(root, out, args.rl_seed)


if __name__ == "__main__":
    main()
