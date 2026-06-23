#!/bin/bash

# Instead of an associative array, use two parallel arrays
MODELS=("whisperx") # "omni") #TODO: add vibevoice when dialogue formatting is implemented
PATHS=(
    "/Users/andreasantos/anaconda3/envs/whisperx/bin/python"
    # "/Users/andreasantos/anaconda3/envs/vibevoice/bin/python"
    "/Users/andreasantos/anaconda3/envs/omni_asr/bin/python"
)

for i in "${!MODELS[@]}"; do
    echo "--- Running benchmark for: ${MODELS[$i]} ---"
    "${PATHS[$i]}" run_benchmark.py --model_filter "${MODELS[$i]}"
    echo "--- Finished ${MODELS[$i]} ---"
done