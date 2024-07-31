#!/bin/bash
CKPT=$1
NAME=$2

python -m eagle.eval.model_vqa_loader \
    --model-path $CKPT \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/${NAME}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python eagle/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/${NAME}.jsonl
