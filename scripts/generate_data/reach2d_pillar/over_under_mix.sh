#!/bin/bash
ENVIRONMENT=Reach2DPillar
N=1000
POLICY=over
POLICY2=under
SEED=0

python ./src/generate_data.py \
    --save_dir ./data/$ENVIRONMENT \
    --environment $ENVIRONMENT \
    --N_trajectories $N \
    --seed $SEED \
    --save_fname $POLICY\_$POLICY2\_mix.pkl \
    --policy $POLICY \
    --policy2 $POLICY2 \
    --perc 0.3
