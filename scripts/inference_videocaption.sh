#!/usr/bin/env bash
MASTER_ADDR=127.0.0.1
MASTER_PORT=2$(($RANDOM % 10))$(($RANDOM % 10))15
WORLD_SIZE=1
RANK=0

GPU_NUM=1
TOTAL_GPU=$((WORLD_SIZE * GPU_NUM))

checkpoint_dir='pretrained_mplug_2.pth'
output_dir='output/videoqa_msrvtt_'${TOTAL_GPU}

mkdir -p ${output_dir}
python -u -m torch.distributed.launch --nproc_per_node=$GPU_NUM \
    --master_addr=$MASTER_ADDR \
	--master_port=$MASTER_PORT \
	--nnodes=$WORLD_SIZE \
	--node_rank=$RANK \
    --use_env \
    video_caption_mplug2.py \
    --config ./configs_video/VideoCaption_msvd_large.yaml \
    --text_encoder bert-large-uncased \
    --text_decoder bert-large-uncased \
    --output_dir ${output_dir} \
    --checkpoint ${checkpoint_dir} \
    --do_two_optim \
    --evaluate \
    --do_amp 2>&1 | tee ${output_dir}/train.log