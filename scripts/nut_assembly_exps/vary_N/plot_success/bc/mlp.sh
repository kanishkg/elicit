#!/bin/bash
ARCH=MLP
CHECKPOINT_FILE=model_best.pt
DATA_SOURCE=oracle
DATE=jan3_drop_layer_hidden20
ENVIRONMENT=Reach2D
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
