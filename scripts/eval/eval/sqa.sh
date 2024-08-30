#!/bin/bash
CKPT=$1
NAME=$2

python -m eagle.eval.model_vqa_science \
    --model-path $CKPT \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/${NAME}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python eagle/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/${NAME}.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/${NAME}_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/${NAME}_result.json
