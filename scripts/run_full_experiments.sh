#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[full_experiments] Running full pipeline"

echo "[full_experiments] Step 1/4: MVTec AD"
bash "$SCRIPT_DIR/pending_experiments.sh" ad

echo "[full_experiments] Step 2/4: MVTec LOCO (Plan A)"
bash "$SCRIPT_DIR/pending_experiments.sh" loco-plan-a

echo "[full_experiments] Step 3/4: MVTec LOCO (SALAD)"
bash "$SCRIPT_DIR/pending_experiments.sh" loco-salad

echo "[full_experiments] Step 4/4: Few-shot"
bash "$SCRIPT_DIR/pending_experiments.sh" fewshot

echo "[full_experiments] Done"
