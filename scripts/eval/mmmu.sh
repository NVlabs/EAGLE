#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
MODEL_CKPT=$1
MODEL_NAME=$2

SAVE_DIR=playground/data/eval/mmmu/${MODEL_NAME}
SPLIT=validation
MMMU_DATA_ROOT=./playground/data/eval/MMMU

python eagle/eval/model_vqa_mmmu.py \
    --model_path ${MODEL_CKPT} \
    --split ${SPLIT} \
    --output_path ${SAVE_DIR}/${SPLIT}_output.json \

output_file=${SAVE_DIR}/${SPLIT}_output.json 
echo "saving model answer at $output_file"

python ./eval_utils/mmmu/main_eval_only.py --output_path ${SAVE_DIR}/${SPLIT}_output.json