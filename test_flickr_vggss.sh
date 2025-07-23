

export CUDA_VISIBLE_DEVICES=0,1,2,3


python test.py \
    --test_data_path /home1/wcl/dataset/Vggss_dataset/vedio_test_frame_audio \
    --test_gt_path  / \
    --model_dir checkpoints \
    --experiment_name exprriment_flicker_10k_valflickr_obeject_0sam_test_0.0004 \
    --testset 'vggss' \
    --alpha 0.4 \
    --gpu 1 \
    --dropout_img 0.9 \
    --dropout_aud 0 \
    --out_dim 256 \
    --num_slots 5 \
    --slot_dim 256 \
    --slot_att_iter 3 \
    --resize_to 224 224 \
    --ISA  \
    --save_visualizations \
    --encoder "dino-vitb-16"

#    --multiprocessing_distributed True

###################### test_data_arg : ######################
# vggss
#--test_data_path /home1/wcl/dataset/Vggss_dataset/vedio_test_frame_audio \
#--test_gt_path 1 \
#--testset 'vggss'

# flickr :
# test_data_path  /home1/wcl/dataset/flcker-process/flicker_test_5k
# test_gt_path /home1/wcl/dataset/flcker-process/flicker_test_5k/Annotations
#--testset 'k_flickr_5k' \

#--test_data_path /home2/wcl/Flicker/flcker-process/test \
#--test_gt_path  /home2/wcl/Flicker/flcker-process/test/Annotations
#--testset 'flickr' \