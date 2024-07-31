#!/bin/bash
CKPT=$1
NAME=$2

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

SPLIT="llava_gqa_testdev_balanced"
GQADIR="./playground/data/eval/gqa/data"
LOCAL_ANSWER_DIR="./playground/data/eval_local_files/gqa"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m eagle.eval.model_vqa_loader \
        --model-path $CKPT \
        --question-file ./playground/data/eval/gqa/$SPLIT.jsonl \
        --image-folder ./playground/data/eval/gqa/data/images \
        --answers-file ${LOCAL_ANSWER_DIR}/$SPLIT/$NAME/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=${LOCAL_ANSWER_DIR}/$SPLIT/$NAME/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${LOCAL_ANSWER_DIR}/$SPLIT/$NAME/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_gqa_for_eval.py --src $output_file --dst ${LOCAL_ANSWER_DIR}/$SPLIT/$NAME/testdev_balanced_predictions.json
absolute_path=$(readlink -f "${LOCAL_ANSWER_DIR}/$SPLIT/$NAME")

cd $GQADIR
# python eval/eval.py --predictions ${LOCAL_ANSWER_DIR}/$SPLIT/$name/{tier}_predictions.json --tier testdev_balanced
python eval.py --predictions ${absolute_path}/{tier}_predictions.json --tier testdev_balanced