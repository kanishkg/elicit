#!/bin/bash
ARCH=MLP
CHECKPOINT_FILE=model_best.pt
DATA_SOURCE=pillar_mix
DATE=jan5_lenient_hidden150_less_lenient
ENVIRONMENT=Reach2DPillar
METHOD=BC
NUM_MODELS=1
SEED=4

python scripts/plot_success_rate.py \
    --arch $ARCH \
    --ckpt_file $CHECKPOINT_FILE \
    --data_source $DATA_SOURCE \
    --date $DATE \
    --environment $ENVIRONMENT \
    --method $METHOD \
    --num_models $NUM_MODELS \
    --seed $SEED
