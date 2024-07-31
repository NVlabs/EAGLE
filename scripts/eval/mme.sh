#!/bin/bash
CKPT=$1
NAME=$2
MME_DATA_ROOT=$(readlink -f "./playground/data/eval/MME")

python -m eagle.eval.model_vqa_loader \
    --model-path $CKPT \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/${NAME}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MME

# python convert_answer_to_mme.py --experiment ${NAME}.jsonl

# cd eval_tool

# python calculation.py --results_dir answers/${NAME}

python convert_answer_to_mme.py --experiment ${MME_DATA_ROOT}/answers/${NAME}.jsonl --data_path ${MME_DATA_ROOT}/MME_Benchmark_release_version

cd eval_tool

python calculation.py --results_dir ${MME_DATA_ROOT}/answers/${NAME}_mme_results
