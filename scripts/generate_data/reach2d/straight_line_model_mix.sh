#!/bin/bash
ENVIRONMENT=Reach2D
N=1000
POLICY=straight_line
POLICY2=model
SEED=0

python ./src/generate_data.py \
    --save_dir ./data/$ENVIRONMENT \
    --model_path ./out/dec22/bc_oracle_reach2d_linear/model_5.pt \
    --arch LinearModel \
    --environment $ENVIRONMENT \
    --N_trajectories $N \
    --seed $SEED \
    --save_fname $POLICY\_$POLICY2\_mix.pkl \
    --policy $POLICY \
    --policy2 $POLICY2 \
    --perc 0.5