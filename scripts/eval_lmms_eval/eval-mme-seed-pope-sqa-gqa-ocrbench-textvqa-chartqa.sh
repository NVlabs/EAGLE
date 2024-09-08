MODEL_PATH=$1
MODEL_NAME=$2
CONV_MODE=$3

accelerate launch --num_processes=8\
           evaluate_lmms_eval.py \
           --model eagle \
           --model_args pretrained=${MODEL_PATH},conv_template=${CONV_MODE} \
           --tasks  mme,seed_bench,pope,scienceqa_img,gqa,ocrbench,textvqa_val,chartqa \
           --batch_size 1 \
           --log_samples \
           --log_samples_suffix ${MODEL_NAME}_mmbench_mathvista_seedbench \
           --output_path ./logs/ 