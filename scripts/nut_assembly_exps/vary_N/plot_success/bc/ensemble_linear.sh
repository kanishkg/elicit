#!/bin/bash
ARCH=LinearModel
CHECKPOINT_FILE=model_4.pt
DATE=dec28
ENVIRONMENT=Reach2D
METHOD=BC
NUM_MODELS=5
SEED=4

python scripts/reach2d_exps/plot_success_rate.py \
    --arch $ARCH \
    --ckpt_file $CHECKPOINT_FILE \
    --date $DATE \
    --environment $ENVIRONMENT \
    --method $METHOD \
    --num_models $NUM_MODELS \
    --seed $SEED
