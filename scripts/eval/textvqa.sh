#!/bin/bash
CKPT=$1
NAME=$2
DATA_ROOT=$(readlink -f "./playground/data/eval/textvqa/")

python -m eagle.eval.model_vqa_loader \
    --model-path $CKPT \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/${NAME}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m eagle.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/${NAME}.jsonl
