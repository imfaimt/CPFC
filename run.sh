#!/bin/bash
EXPERIMENT=exprriment_flicker_144k_valflickr_obeject_10sam_warmup10_test_0.000024
rm -rf checkpoints/$EXPERIMENT

export CUDA_VISIBLE_DEVICES=1,2

torchrun --nproc_per_node=2 \
    --master_port=12345 \
    cpcf_train_with_viz.py \
    --workers 8 \
    --train_data_path /home1/wcl/dataset/flcker-process/flicker_144k \
    --test_data_path /home1/wcl/dataset/flcker-process/test \
    --test_gt_path /home1/wcl/dataset/flcker-process/test/Annotations \
    --experiment_name $EXPERIMENT \
    --trainset 'flickr_144k' \
    --testset 'flickr' \
    --encoder "dino-vitb-16" \
    --epochs 30 \
    --batch_size 50 \
    --warmup 20 \
    --learning_rate 0.000024 \
    --loss1_weight 1 \
    --loss2_weight 10 \
    --loss3_weight 1 \
    --slot_dim 256 \
    --ISA \
    --resize_to 224 224 \
    --num_slots 5 \
    --slot_att_iter 3

# pip install -r requirements.txt