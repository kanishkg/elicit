#!/bin/bash
ENVIRONMENT=NutAssembly
N=30
POLICY=user
SEED=0

python ./src/generate_data.py \
    --save_dir ./data/$ENVIRONMENT \
    --environment $ENVIRONMENT \
    --N_trajectories $N \
    --seed $SEED \
    --save_fname $POLICY.pkl \
    --policy $POLICY \
    --robosuite
