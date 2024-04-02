#!/usr/bin/env bash
# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------


set -x
set -o pipefail



DATA_DIR=DanceTrack
DATA_SPLIT=val

EXP_NAME=tracker_sam_occlusion_discard_v2
args=$(cat configs/motrv2_sam_occlusion_discard.args)
CUDA_VISIBLE_DEVICES=0 python3 submit_dance.py ${args} --exp_name outputs/${EXP_NAME}-${DATA_SPLIT} --resume $1 --data_dir ${DATA_DIR}/${DATA_SPLIT}


