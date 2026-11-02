#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${AST_DATA:?Set AST_DATA to the data directory}"
export AST_OUTPUT="${AST_OUTPUT:-outputs}" PYTHONPATH="${PYTHONPATH:-}:$PWD"
JOBS="${JOBS:-1}"

studies() {
    python experiments/study1_sar.py --scores expert
    python experiments/study1_sar.py --scores ensemble
    python experiments/study1_agreement.py
    python experiments/study2_maptask.py
    python experiments/study2_multiwoz.py
    python experiments/study2_annomi.py
    python experiments/study3_minigrid.py
    python experiments/study3_crafter.py
    python experiments/study3_metaworld.py
    python experiments/study3_boir.py
}

rl() {
    python experiments/study4_train.py --jobs "$JOBS"
    python experiments/study4_evaluate.py "$AST_OUTPUT/study4_checkpoints"
}

case "${1:-all}" in
    studies) studies ;;
    rl) rl ;;
    figures) ;;
    all) studies; rl ;;
    *) echo "usage: $0 [all|studies|rl|figures]"; exit 1 ;;
esac
python experiments/figures.py --outputs "$AST_OUTPUT"
python experiments/check.py --outputs "$AST_OUTPUT"
