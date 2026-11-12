#!/bin/sh
# DRESS REHEARSAL of the whole final-test harness on the 300 VAL routes laid out like data/eval_frozen
# (mock sealed answers + mock MANIFEST). Stage 1 (fast subset of arms) -> validity check -> stage 2
# -> checks against the stored val results (v0-hist must reproduce EXACTLY). Everything is written
# under runs/final/dress/; the real eval folder, real predictions and results/ are not touched.
#   sh scripts/final_dress_rehearsal.sh
set -eu
TR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$TR"
D="$TR/runs/final/dress"
export TR_EVAL_DIR="$D/eval_frozen" TR_FINAL_DIR="$D" TR_RESULTS_DIR="$D/results" TR_PREREG_DOC="$D/PREREGISTRATION.md"
export OMP_NUM_THREADS=1
P=".venv/bin/python -I"
step() { echo "== dress: $1 $(date -u +%FT%TZ)"; shift; "$@"; }
step setup        $P scripts/final_dress_rehearsal.py setup
step models       $P scripts/final_predict.py models --arms v0hist,v1,v1_exact --workers 3
step rules        $P scripts/final_predict.py rules --arms R2,S3 --workers 3
step jpt-prepare  $P scripts/final_predict.py jpt-prepare
step jpt-apply    $P scripts/final_predict.py jpt-apply
step opus3-rep1   $P scripts/final_predict.py opus --run 3 --rep 1
step opus3-rep2   $P scripts/final_predict.py opus --run 3 --rep 2
step opus-model   $P scripts/final_predict.py zone-rich-collect --out "$D/zone_rich_dev_A" --solver greedy
step validity     $P scripts/final_validity_check.py
step score        $P scripts/final_score.py --workers 3
step check        $P scripts/final_dress_rehearsal.py check
echo "== dress: done $(date -u +%FT%TZ)"
