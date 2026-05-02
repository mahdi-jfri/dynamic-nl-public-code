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

DYNAMIC_DATASETS_STR="${DYNAMIC_DATASETS:-arxiv products sbm500k}"
DYNAMIC_PHASES_STR="${DYNAMIC_PHASES:-main eps alpha}"
N_SEEDS_MAIN="${N_SEEDS_MAIN:-10}"
N_SEEDS_SWEEP="${N_SEEDS_SWEEP:-2}"
EPS="${EPS:-1e-7}"
EPS_VALUES_STR="${EPS_VALUES_STR:-1e-3 1e-4 1e-5 1e-6 1e-7}"

IFS=' ' read -ra DYNAMIC_DATASETS <<< "${DYNAMIC_DATASETS_STR}"
IFS=' ' read -ra DYNAMIC_PHASES <<< "${DYNAMIC_PHASES_STR}"
IFS=' ' read -ra EPS_VALUES <<< "${EPS_VALUES_STR}"

tag_value() {
  echo "$1" | tr '.-' 'pm'
}

set_dataset_args() {
  local dataset_name="$1"
  case "${dataset_name}" in
    arxiv)
      DATASET_KEY="arxiv"
      DATASET_ARG="arxiv"
      DEFAULT_MLP_EPOCHS=1000
      DEFAULT_ALPHA_VALUES_STR="0.001 0.01 0.1 0.2 0.3 0.5 0.7"
      COMMON_PREFIX=(--dataset arxiv --feat-scale 3 -t rowsum --beta 0.5)
      MAIN_ALPHA="0.1"
      COMMON_SUFFIX=(--mlp-hidden 1024 --mlp-num-layers 4)
      ;;
    products)
      DATASET_KEY="products"
      DATASET_ARG="products"
      DEFAULT_MLP_EPOCHS=1000
      DEFAULT_ALPHA_VALUES_STR="0.01 0.1 0.2 0.3 0.5 0.7"
      COMMON_PREFIX=(--dataset products --batch-size 512 --feat-scale 3 -t rowsum --beta 0.5)
      MAIN_ALPHA="0.1"
      COMMON_SUFFIX=(--mlp-hidden 1024 --mlp-num-layers 4 --mlp-dropout 0.5)
      ;;
    sbm500k | sbm_500k | sbm-500k)
      DATASET_KEY="sbm_500k"
      DATASET_ARG="sbm-500k"
      DEFAULT_MLP_EPOCHS=200
      DEFAULT_ALPHA_VALUES_STR="0.001 0.01 0.1 0.2 0.3 0.5 0.7"
      COMMON_PREFIX=(--dataset sbm-500k --batch-size 1024 --feat-scale 3 -t rowsum --beta 0.5)
      MAIN_ALPHA="0.001"
      COMMON_SUFFIX=(--mlp-hidden 1024 --mlp-num-layers 2 --mlp-dropout 0.1 --mlp-lr 0.01)
      ;;
    *)
      echo "Unknown dynamic dataset: ${dataset_name}" >&2
      exit 2
      ;;
  esac
  MLP_EPOCHS_EFFECTIVE="${MLP_EPOCHS:-${DEFAULT_MLP_EPOCHS}}"
  ALPHA_VALUES_EFFECTIVE="${ALPHA_VALUES_STR:-${DEFAULT_ALPHA_VALUES_STR}}"
}

launch_dynamic() {
  local out_stem="$1"; shift
  local mode="$1"; shift
  local seed="$1"; shift
  local step_arg_string="$1"; shift
  local -a step_args=()
  read -r -a step_args <<< "${step_arg_string}"

  mkdir -p "$(dirname "${out_stem}")"
  echo "Running ${out_stem}.json"
  "${PY}" -u -m dynamic.run_experiment \
    --seed "${seed}" \
    --mode "${mode}" \
    "${step_args[@]}" \
    "$@" \
    --out "${out_stem}.json" \
    > "${out_stem}.log" 2>&1
}

run_main_phase() {
  local out_base="outputs/dynamic/${DATASET_KEY}"
  mkdir -p "${out_base}" "${out_base}/single" "${out_base}/from_scratch"
  local -a common=("${COMMON_PREFIX[@]}" --alpha "${MAIN_ALPHA}" --eps "${EPS}" --mlp-epochs "${MLP_EPOCHS_EFFECTIVE}" "${COMMON_SUFFIX[@]}")

  for seed in $(seq 0 $((N_SEEDS_MAIN - 1))); do
    launch_dynamic "${out_base}/nl${seed}" dynamic_batched "${seed}" "--step-fn htanh --step-param 2.5" "${common[@]}"
    launch_dynamic "${out_base}/single/nl${seed}" dynamic "${seed}" "--step-fn htanh --step-param 2.5" "${common[@]}"
    launch_dynamic "${out_base}/from_scratch/nl${seed}" from_scratch "${seed}" "--step-fn htanh --step-param 2.5" "${common[@]}"
    launch_dynamic "${out_base}/ours${seed}" dynamic_batched "${seed}" "--step-fn none" "${common[@]}"
    launch_dynamic "${out_base}/single/ours${seed}" dynamic "${seed}" "--step-fn none" "${common[@]}"
    launch_dynamic "${out_base}/from_scratch/ours${seed}" from_scratch "${seed}" "--step-fn none" "${common[@]}"
  done
}

run_eps_phase() {
  local out_base="outputs/dynamic/${DATASET_KEY}_eps"
  mkdir -p \
    "${out_base}/batched/nl" "${out_base}/single/nl" "${out_base}/from_scratch/nl" \
    "${out_base}/batched/linear" "${out_base}/single/linear" "${out_base}/from_scratch/linear"

  for seed in $(seq 0 $((N_SEEDS_SWEEP - 1))); do
    for eps_value in "${EPS_VALUES[@]}"; do
      eps_tag="$(tag_value "${eps_value}")"
      local -a common=("${COMMON_PREFIX[@]}" --alpha "${MAIN_ALPHA}" --eps "${eps_value}" --mlp-epochs "${MLP_EPOCHS_EFFECTIVE}" "${COMMON_SUFFIX[@]}")
      launch_dynamic "${out_base}/batched/nl/eps${eps_tag}_seed${seed}" dynamic_batched "${seed}" "--step-fn htanh --step-param 2.5" "${common[@]}"
      launch_dynamic "${out_base}/single/nl/eps${eps_tag}_seed${seed}" dynamic "${seed}" "--step-fn htanh --step-param 2.5" "${common[@]}"
      launch_dynamic "${out_base}/from_scratch/nl/eps${eps_tag}_seed${seed}" from_scratch "${seed}" "--step-fn htanh --step-param 2.5" "${common[@]}"
      launch_dynamic "${out_base}/batched/linear/eps${eps_tag}_seed${seed}" dynamic_batched "${seed}" "--step-fn none" "${common[@]}"
      launch_dynamic "${out_base}/single/linear/eps${eps_tag}_seed${seed}" dynamic "${seed}" "--step-fn none" "${common[@]}"
      launch_dynamic "${out_base}/from_scratch/linear/eps${eps_tag}_seed${seed}" from_scratch "${seed}" "--step-fn none" "${common[@]}"
    done
  done
}

run_alpha_phase() {
  local out_base="outputs/dynamic/${DATASET_KEY}_alpha"
  mkdir -p \
    "${out_base}/batched/nl" "${out_base}/single/nl" "${out_base}/from_scratch/nl" \
    "${out_base}/batched/linear" "${out_base}/single/linear" "${out_base}/from_scratch/linear"

  IFS=' ' read -ra ALPHA_VALUES <<< "${ALPHA_VALUES_EFFECTIVE}"
  for seed in $(seq 0 $((N_SEEDS_SWEEP - 1))); do
    for alpha_value in "${ALPHA_VALUES[@]}"; do
      alpha_tag="$(tag_value "${alpha_value}")"
      local -a common=("${COMMON_PREFIX[@]}" --alpha "${alpha_value}" --eps "${EPS}" --mlp-epochs "${MLP_EPOCHS_EFFECTIVE}" "${COMMON_SUFFIX[@]}")
      launch_dynamic "${out_base}/batched/nl/alpha${alpha_tag}_seed${seed}" dynamic_batched "${seed}" "--step-fn htanh --step-param 2.5" "${common[@]}"
      launch_dynamic "${out_base}/single/nl/alpha${alpha_tag}_seed${seed}" dynamic "${seed}" "--step-fn htanh --step-param 2.5" "${common[@]}"
      launch_dynamic "${out_base}/from_scratch/nl/alpha${alpha_tag}_seed${seed}" from_scratch "${seed}" "--step-fn htanh --step-param 2.5" "${common[@]}"
      launch_dynamic "${out_base}/batched/linear/alpha${alpha_tag}_seed${seed}" dynamic_batched "${seed}" "--step-fn none" "${common[@]}"
      launch_dynamic "${out_base}/single/linear/alpha${alpha_tag}_seed${seed}" dynamic "${seed}" "--step-fn none" "${common[@]}"
      launch_dynamic "${out_base}/from_scratch/linear/alpha${alpha_tag}_seed${seed}" from_scratch "${seed}" "--step-fn none" "${common[@]}"
    done
  done
}

for dataset in "${DYNAMIC_DATASETS[@]}"; do
  set_dataset_args "${dataset}"
  echo "Dataset ${DATASET_ARG}"
  for phase in "${DYNAMIC_PHASES[@]}"; do
    case "${phase}" in
      main) run_main_phase ;;
      eps) run_eps_phase ;;
      alpha) run_alpha_phase ;;
      *) echo "Unknown phase: ${phase}" >&2; exit 2 ;;
    esac
  done
done
