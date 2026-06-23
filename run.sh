#!/bin/bash
declare -A MODEL_ENVS=(
  [whisperx]="whisperx"
  [omni]="omni_asr"
  [vibevoice]="vibevoice"   
)

for model in "${!MODEL_ENVS[@]}"; do
    echo "--- Running benchmark for: ${model} ---"
    if conda run -n "${MODEL_ENVS[$model]}" python run_benchmark.py --model_filter "$model"; then
        echo "--- Finished ${model} ---"
    else
        echo "!!! ${model} failed (exit $?) !!!" >&2
        FAILED+=("$model")
    fi
done

if (( ${#FAILED[@]} )); then
    echo "Failed models: ${FAILED[*]}" >&2
    exit 1
fi