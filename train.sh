#!/bin/bash
# NODE_COUNT=1
# RANK=0
# MASTER_PORT=29500
# MASTER_ADDR="localhost"
# GPUS=`nvidia-smi -L | wc -l`

OUTPUT_DIR='OUTPUT/train_mouse'
MODEL_PATH='OUTPUT/mvd_s_from_b_ckpt_399.pth'
DATA_PATH="data/mouse/HOM Mice F.2632_HOM 12 Days post tre/12 Days post tre"
DATA_ROOT="data/mouse/HOM Mice F.2632_HOM 12 Days post tre/12 Days post tre"

python run_class_finetuning.py \
    --model vit_small_patch16_224 \
    --data_set Kinetics-400 --nb_classes 2 \
    --data_path "${DATA_PATH}" \
    --data_root "${DATA_ROOT}" \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --input_size 224 --short_side_size 224 \
    --opt adamw --opt_betas 0.9 0.999 --weight_decay 0.001 \
    --batch_size 2 --update_freq 2 --num_sample 2 \
    --save_ckpt_freq 5 --no_save_best_ckpt \
    --num_frames 16 --sampling_rate 4 \
    --lr 5e-4 --epochs 100 \
    --dist_eval --test_num_segment 5 --test_num_crop 3 \
    --enable_deepspeed