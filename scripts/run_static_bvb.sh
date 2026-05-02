#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd)"
cd "${REPO_DIR}"

if [[ -n "${PYTHON:-}" ]]; then
  PY="${PYTHON}"
elif [[ -x "${REPO_DIR}/.venv/bin/python" ]]; then
  PY="${REPO_DIR}/.venv/bin/python"
else
  PY="python3"
fi

STEPS_LIST="${STEPS_LIST:-50}"
SPLITS_STR="${SPLITS:-0 1 2 3 4 5 6 7 8 9}"
BVB_DATASETS_STR="${BVB_DATASETS:-actor pubmed citeseer cornell texas wisconsin squirrel chameleon cora chameleon_new squirrel_new}"

IFS=' ' read -ra SPLITS_ARR <<< "${SPLITS_STR}"
IFS=' ' read -ra DATASETS <<< "${BVB_DATASETS_STR}"

for dataset in "${DATASETS[@]}"; do
  ts="$(date +%Y-%m-%d_%H-%M-%S)"
  out_dir="outputs/${ts}_${dataset}_union"
  log="${out_dir}/bvb_${dataset}.log"
  mkdir -p "${out_dir}"

  "${PY}" -u bvb.py \
    --dataset "${dataset}" \
    --log "${log}" \
    --normalizations std \
    --feat-scales 1 3 5 \
    --alphas 0.01 0.1 0.2 0.3 0.5 0.7 \
    --betas 0.5 1.0 \
    --steps-list "${STEPS_LIST}" \
    --step-fns none tanh clamp leaky relu sigmoid \
      stanh0.5 stanh1 stanh1.5 stanh2 stanh3 stanh4 stanh5 stanh6 stanh8 \
      htanh0.3 htanh0.5 htanh0.7 htanh1 htanh1.5 htanh2 htanh2.5 \
      shtanh0.5 shtanh-0.5 shtanh-0.9 shtanh-1.2 stanh3 \
    --classifiers linear mlp --hiddens 512 --dropouts 0.5 \
    --lrs 0.001 0.01 0.1 0.5 \
    --wds 5e-4 \
    --splits "${SPLITS_ARR[@]}" \
    --seeds 0 1 2 3 4 \
    --epochs 400

  echo "${dataset} BVB sweep complete: ${log}"
done
