#!/bin/bash

# torchrun \
# --nnodes 1 \
# --nproc_per_node 4 \
# --master_port 29711 \
# /home/yb107/cvit-work/cvpr2025/SAT/train.py \
# --log_dir '/home/yb107/cvit-work/cvpr2025/SAT/log' \
# --name 'nano_base_SAT_model_train' \
# --vision_backbone 'UNET' \
# --deep_supervision True \
# --save_large_interval 100 \
# --save_small_interval 100 \
# --log_step_interval 100 \
# --step_num 260000 100000 \
# --warmup 10000 10000 \
# --lr 1e-4 1e-5 \
# --accumulate_grad_interval 1 \
# --datasets_jsonl '/home/yb107/cvit-work/cvpr2025/SAT/data/challenge_data/train_10percent_fixed.jsonl' \
# --dataset_config '/home/yb107/cvit-work/cvpr2025/SAT/data/dataset_config/cvpr25.json' \
# --text_prompts_json '/home/yb107/cvit-work/cvpr2025/SAT/data/challenge_data/CVPR25_TextSegFMData_with_class.json' \
# --text_encoder 'ours' \
# --text_encoder_checkpoint '/home/ld258/cvpr_seg_2025/models/text_encoder_cvpr25_v0.pth' \
# --text_encoder_partial_load True \
# --open_bert_layer 12 \
# --open_modality_embed False \
# --num_workers 8 \
# --max_queries 32 \
# --crop_size 288 288 96 \
# --patch_size 32 32 32 \
# --batchsize_3d 1  \
# --allow_repeat True \
# --pin_memory False \
# --nnUNet_aug True

# Chnage three places nproc_per_node, name, checkpoint and dataset_jsonl
# --resume True \
# --checkpoint '/cachedata/yb107/results/logs/nano_US_SAT_model/checkpoint/latest_step.pth' \

torchrun \
--nnodes 1 \
--nproc_per_node 1 \
--master_port 29716 \
/home/yb107/cvit-work/cvpr2025/SAT/train.py \
--log_dir '/cachedata/yb107/results/logs' \
--name 'nano_PET_SAT_model' \
--vision_backbone 'UNET' \
--deep_supervision True \
--save_large_interval 1000 \
--save_small_interval 100 \
--log_step_interval 100 \
--step_num 260000 100000 \
--warmup 10000 10000 \
--lr 1e-4 1e-5 \
--accumulate_grad_interval 1 \
--datasets_jsonl '/cachedata/yb107/data/samples/PET.jsonl' \
--dataset_config '/cachedata/yb107/data/dataset_config/cvpr25.json' \
--text_prompts_json '/cachedata/yb107/data/CVPR25_TextSegFMData_with_class.json' \
--text_encoder 'ours' \
--text_encoder_checkpoint '/cachedata/yb107/models/text_encoder_cvpr25_v0.pth' \
--text_encoder_partial_load True \
--open_bert_layer 12 \
--open_modality_embed False \
--num_workers 8 \
--max_queries 32 \
--crop_size 288 288 96 \
--patch_size 32 32 32 \
--batchsize_3d 1  \
--allow_repeat True \
--pin_memory False \
--nnUNet_aug True




# --step_num 260000 100000 \
# --warmup 10000 10000 \
# --max_queries 32 \
# --crop_size 288 288 96 \
# --patch_size 32 32 32 \
# --batchsize_3d 2  \
# --datasets_jsonl '/home/yb107/cvit-work/cvpr2025/SAT/data/challenge_data/train_45.jsonl' \

# torchrun \
# --nnodes 1 \
# --nproc_per_node 1 \
# --master_port 29712 \
# /home/yb107/cvit-work/cvpr2025/SAT/train.py \
# --log_dir '/home/yb107/cvit-work/cvpr2025/SAT/log' \
# --name 'code-explain_test' \
# --vision_backbone 'UNET' \
# --deep_supervision True \
# --save_large_interval 100 \
# --save_small_interval 100 \
# --log_step_interval 100 \
# --step_num 260000 100000 \
# --warmup 10000 10000 \
# --lr 1e-4 1e-5 \
# --accumulate_grad_interval 1 \
# --datasets_jsonl '/home/yb107/cvit-work/cvpr2025/SAT/data/challenge_data/train_30.jsonl' \
# --dataset_config '/home/yb107/cvit-work/cvpr2025/SAT/data/dataset_config/cvpr25.json' \
# --text_prompts_json '/home/yb107/cvit-work/cvpr2025/SAT/data/challenge_data/CVPR25_TextSegFMData_with_class.json' \
# --text_encoder 'ours' \
# --text_encoder_checkpoint '/home/ld258/cvpr_seg_2025/models/text_encoder_cvpr25_v0.pth' \
# --text_encoder_partial_load True \
# --open_bert_layer 12 \
# --open_modality_embed False \
# --num_workers 8 \
# --max_queries 32 \
# --crop_size 288 288 96 \
# --patch_size 32 32 32 \
# --batchsize_3d 1  \
# --allow_repeat True \
# --pin_memory False \
# --nnUNet_aug True


# Also see line 58 in Maskformer.py
