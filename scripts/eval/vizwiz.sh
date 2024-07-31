#!/bin/bash
CKPT=$1
NAME=$2
DATA_ROOT=$(readlink -f "./playground/data/eval/vizwiz")
LOCAL_ANSWER_DIR="./playground/data/eval_local_files/vizwiz"

python -m eagle.eval.model_vqa_loader \
    --model-path $CKPT \
    --question-file $DATA_ROOT/llava_test.jsonl \
    --image-folder $DATA_ROOT/test \
    --answers-file $LOCAL_ANSWER_DIR/$NAME/$NAME.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file $DATA_ROOT/llava_test.jsonl \
    --result-file $LOCAL_ANSWER_DIR/$NAME/$NAME.jsonl \
    --result-upload-file $LOCAL_ANSWER_DIR/$NAME/answers_upload/vizwiz_test_$NAME.json
