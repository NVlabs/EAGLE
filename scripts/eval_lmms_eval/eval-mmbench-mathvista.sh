MODEL_PATH=$1
MODEL_NAME=$2
CONV_MODE=$3

accelerate launch --num_processes=8\
           evaluate_lmms_eval.py \
           --model eagle \
           --model_args pretrained=${MODEL_PATH},conv_template=${CONV_MODE} \
           --tasks mmbench_en_dev,mathvista_testmini \
           --batch_size 1 \
           --log_samples \
           --log_samples_suffix ${MODEL_NAME}_mmbench_mathvista \
           --output_path ./logs/ 