MODEL_PATH=$1
MODEL_NAME=$2
CONV_MODE=$3

accelerate launch --num_processes=8\
           evaluate_lmms_eval.py \
           --model eagle \
           --model_args pretrained=${MODEL_PATH},conv_template=${CONV_MODE} \
           --tasks  vizwiz_vqa_test,vqav2_test \
           --batch_size 1 \
           --log_samples \
           --log_samples_suffix ${MODEL_NAME}_vizwiz_vqav2 \
           --output_path ./logs/ 