# Acquisition-State Telemetry

Code for *Task-referenced telemetry distinguishes information acquisition from activity in autonomous systems*.

Acquisition-State Telemetry (AST) measures, at every exchange, how an autonomous system is acquiring the information
its task requires. A task binding maps each observation to resolution scores over a declared task schema and to a
unit-norm evidence embedding. A frozen monitor turns this stream into two telemetry channels: **progress**, the
residual acquisition value of each task coordinate, and **friction**, recurrent evidence that yields no gain while
the task remains unresolved. One monitor configuration is applied, unchanged, to dialogue, grid exploration,
capability unlocking, robotic manipulation, Bayesian intent inference and reinforcement-learning control.

```python
from asttel.monitor import Monitor

monitor = Monitor(schema=["location", "time", "description"])
progress, friction, gains = monitor.step(embedding, scores={"location": 0.4}, target="location")
```

## Repository

```
asttel/                     monitor and bindings
  params.py                 frozen parameters and decision thresholds
  monitor.py                progress and friction channels
  dialogue.py               dialogue session runner and session metrics
  events.py                 stall detection and next-gain prediction metrics
  stats.py                  sign and binomial tests
  trajectory.py             per-exchange trajectory files
  data.py                   data and output paths
  bindings/                 MiniGrid, Crafter, MetaWorld and BOIR bindings
  rl/env.py                 interview environment and the five policy variants
  rl/evaluate.py            policy rollouts and bootstrap intervals
experiments/                one script per study
  study1_sar.py             Study 1: SAR inquiry regimes (expert or LLM-ensemble scores)
  study1_agreement.py       Study 1: agreement of the LLM scorers
  study2_maptask.py         Study 2: HCRC Map Task
  study2_multiwoz.py        Study 2: MultiWOZ
  study2_annomi.py          Study 2: AnnoMI
  study3_minigrid.py        Study 3: MiniGrid
  study3_crafter.py         Study 3: Crafter
  study3_metaworld.py       Study 3: MetaWorld
  study3_boir.py            Study 3: BOIR
  study4_train.py           Study 4: PPO training
  study4_evaluate.py        Study 4: PPO evaluation and Table 1
  figures.py                paper figures and one figure per study
  check.py                  comparison of all results with reference/
  prepare_multiwoz.py       builds multiwoz/sessions.json from MultiWOZ 2.4
tools/
  embed.py                  qwen3-embedding vectors (Ollama)
  score_sar.py              five-model LLM scores for Study 1 (Ollama)
reference/                  reference summaries of every study
tests/                      unit tests
reproduce.sh                runs every study, the figures and the check
```

The data are distributed separately. Every script reads from the directory named by `AST_DATA` and writes to
`AST_OUTPUT` (default `outputs/`). The data directory has its own `README.md`, licences and checksums.

## Installation

Python 3.10 or later.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[envs,rl,test]"
export AST_DATA=/path/to/ast-data
pytest
```

## Reproduction

```bash
./reproduce.sh studies          # Studies 1-3, about 10 min on one core
JOBS=16 ./reproduce.sh rl       # Study 4, about 1 h on 16 cores
./reproduce.sh                  # everything
```

Each command ends by drawing the figures and comparing every result with `reference/`. All computation runs on CPU;
the stored embeddings and LLM scores make language models unnecessary.

| Study | Script | Output | Paper |
|---|---|---|---|
| 1 | `study1_sar.py --scores expert` | `study1_sar_expert/` | Fig. 3, Supplementary Table 2 |
| 1 | `study1_sar.py --scores ensemble` | `study1_sar_ensemble/` | Supplementary Table 2 |
| 1 | `study1_agreement.py` | `study1_agreement/` | Supplementary Note 2 |
| 2 | `study2_maptask.py` | `study2_maptask/` | Results |
| 2 | `study2_multiwoz.py` | `study2_multiwoz/` | Results |
| 2 | `study2_annomi.py` | `study2_annomi/` | Supplementary Table 3 |
| 3 | `study3_minigrid.py` | `study3_minigrid/` | Extended Data Table 2, Supplementary Table 4 |
| 3 | `study3_crafter.py` | `study3_crafter/` | Extended Data Table 2, Supplementary Table 4 |
| 3 | `study3_metaworld.py` | `study3_metaworld/` | Fig. 4, Extended Data Table 2, Supplementary Table 4 |
| 3 | `study3_boir.py` | `study3_boir/` | Supplementary Table 5 |
| 4 | `study4_train.py --jobs N` | `study4_checkpoints/` | Table 1 |
| 4 | `study4_evaluate.py outputs/study4_checkpoints` | `study4/` | Table 1, Fig. 5 |
| – | `figures.py` | `figures/` | Figs 3–5, supplementary figures |

Every study writes `summary.json` and one trajectory file per session or run. The first line of a trajectory holds
the run metadata and task schema; each following line holds one exchange: target coordinate, resolution state,
realized gains, progress and friction. `figures.py --trajectories <files>` draws any of them.

Study 4 trains five variants under two termination conditions, with ten seeds on three corpora (300 policies). The
operational-cost weight of the Coupled-Blind variant is set before training so that its cost and the friction
penalty have equal mean magnitude under a uniform-random policy; the value is written to
`study4_checkpoints/rho.json`. Training is single-threaded per process and seeded, so results are independent of
`JOBS`.

## Regenerating model outputs

With a local Ollama server:

```bash
pip install -e ".[tools]"
python tools/embed.py sar maptask multiwoz annomi
python tools/score_sar.py
```

## Licence

Code: MIT. Data: see the data directory.
