#!/bin/bash
ENVIRONMENT=Reach2D
N=1000
POLICY=up_right
SEED=0

python ./src/generate_data.py \
    --save_dir ./data/$ENVIRONMENT \
    --environment $ENVIRONMENT \
    --N_trajectories $N \
    --seed $SEED \
    --save_fname $POLICY.pkl \
    --policy $POLICY 
