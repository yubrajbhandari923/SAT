#!/bin/bash

# evaluate validation subset
# --text_encoder_checkpoint '/home/yb107/cvit-work/cvpr2025/SAT/log/nano_base_SAT_model_train/checkpoint/text_encoder_step_20300.pth' \

# Changes in the following lines:
# rcd_dir (29438), rcd_file, checkpoint, datasets_jsonl
torchrun \
--nnodes 1 \
--nproc_per_node 1 \
--master_port 29440 \
/home/yb107/cvit-work/cvpr2025/SAT/evaluate.py \
--rcd_dir '/home/yb107/cvit-work/cvpr2025/SAT/data/validation_runs/nano_' \
--rcd_file 'baseline_nano_visualize' \
--visualization True \
--deep_supervision False \
--text_prompts_json '/home/yb107/cvit-work/cvpr2025/SAT/data/challenge_data/CVPR25_TextSegFMData_with_class.json' \
--datasets_jsonl '/home/yb107/cvit-work/cvpr2025/SAT/data/challenge_data/val_subsets/visualize.jsonl' \
--online_crop True \
--crop_size 288 288 96 \
--vision_backbone 'UNET' \
--checkpoint '/cachedata/yb107/models/nano_cvpr25_v0.pth' \
--text_encoder_checkpoint '/cachedata/yb107/models/text_encoder_cvpr25_v0.pth' \
--partial_load True \
--text_encoder 'ours' \
--batchsize_3d 1 \
--max_queries 256 \
--pin_memory False \
--num_workers 8 \
--dice True \
--nsd True