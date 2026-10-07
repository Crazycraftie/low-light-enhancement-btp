#!/usr/bin/env bash
# Run the whole Day-1 evaluation in one go.
#
#   1. Classical baselines (HE, CLAHE, Gamma) on every dataset that exists in data/.
#   2. evaluate.py for: Input (reference), classical methods, and every deep
#      method folder that exists under results/<method>/<dataset>/.
#
# Anything missing (dataset not downloaded, method not run yet) is SKIPPED with
# a message, so you can re-run this script any time new results arrive.
# results/results.csv is updated in place (one row per method+dataset).
#
# Usage (from the repo root, inside the llie env):
#   conda activate llie
#   bash scripts/eval_all.sh            # classical + evaluate everything
#   bash scripts/eval_all.sh --no-classical   # only evaluate (skip regenerating classical outputs)

set -u
cd "$(dirname "$0")/.."   # always run from the repo root

RUN_CLASSICAL=1
[[ "${1:-}" == "--no-classical" ]] && RUN_CLASSICAL=0

# ---- Dataset table:  name | low-light input folder | ground-truth folder ("" = unpaired, NIQE only)
DATASETS=(
  "LOLv1|data/LOLv1/Test/input|data/LOLv1/Test/target"
  "LOLv2-real|data/LOLv2/Real_captured/Test/Low|data/LOLv2/Real_captured/Test/Normal"
  "LOLv2-syn|data/LOLv2/Synthetic/Test/Low|data/LOLv2/Synthetic/Test/Normal"
  "LIME|data/unpaired/LIME|"
  "DICM|data/unpaired/DICM|"
  "MEF|data/unpaired/MEF|"
)

# ---- Method table:  name in results.csv | output folder pattern (DATASET is replaced)
# Deep methods follow the rule results/<method>/<dataset>/.
METHODS=(
  "HE|results/classical/he/DATASET"
  "CLAHE|results/classical/clahe/DATASET"
  "Gamma|results/classical/gamma/DATASET"
  "Zero-DCE|results/zerodce/DATASET"
  "SCI|results/sci/DATASET"
  "SNR-Aware|results/snr_aware/DATASET"
  "SNR-Aware (released)|results/snr_aware_released/DATASET"   # authors' released images (scripts/import_released.py)
  "LLFormer|results/llformer/DATASET"
  "Retinexformer|results/retinexformer/DATASET"
  "GSAD|results/gsad/DATASET"                                          # real output (main numbers)
  "GSAD (GT-mean, authors protocol)|results/gsad_gtmean/DATASET"       # uses GT brightness: reference only
  "Retinexformer (my training)|results/retinexformer_mine/DATASET"   # LOLv1 + unpaired only (LOLv2-real overlaps LOLv1 train)
  "Retinexformer (my training; best-on-test ckpt)|results/retinexformer_mine_best/DATASET"
)

# Helper: does a folder exist AND contain at least one file?
has_images() { [[ -d "$1" ]] && [[ -n "$(ls -A "$1" 2>/dev/null | grep -v '^\.')" ]]; }

# Helper: run evaluate.py in paired (gt given) or NIQE (gt empty) mode.
evaluate() {  # $1=pred folder  $2=gt folder or ""  $3=method  $4=dataset
  if [[ -n "$2" ]]; then
    python scripts/evaluate.py --pred "$1" --gt "$2" --method "$3" --dataset "$4"
  else
    python scripts/evaluate.py --pred "$1" --method "$3" --dataset "$4" --niqe
  fi
}

for entry in "${DATASETS[@]}"; do
  IFS='|' read -r ds input gt <<< "$entry"
  echo "=================== $ds ==================="

  if ! has_images "$input"; then
    echo "  [skip] $ds not downloaded ($input is empty/missing)"
    continue
  fi

  # 1) Classical baselines
  if [[ $RUN_CLASSICAL == 1 ]]; then
    python scripts/classical.py --input "$input" --dataset "$ds"
  fi

  # 2) Input row = the raw dark image scored as if it were a prediction.
  #    This is the "do nothing" reference every method must beat.
  evaluate "$input" "$gt" "Input" "$ds"

  # 3) Every method whose output folder exists
  for m in "${METHODS[@]}"; do
    IFS='|' read -r name pattern <<< "$m"
    pred="${pattern//DATASET/$ds}"
    if has_images "$pred"; then
      evaluate "$pred" "$gt" "$name" "$ds"
    else
      echo "  [skip] $name: no outputs in $pred"
    fi
  done
done

echo
echo "Summary table: results/results.csv"
column -s, -t < results/results.csv 2>/dev/null || cat results/results.csv
