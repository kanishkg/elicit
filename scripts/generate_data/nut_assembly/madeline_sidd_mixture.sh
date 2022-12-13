#!/bin/bash
ENVIRONMENT=NutAssembly
N=30
PERCS=(0.5 0.6 0.7 0.8 0.9)
POLICY=madeline
POLICY2=sidd
SEED=0

for PERC in "${PERCS[@]}"
do
    python ./src/generate_data.py \
        --save_dir ./data/$ENVIRONMENT \
        --environment $ENVIRONMENT \
        --N_trajectories $N \
        --seed $SEED \
        --save_fname $POLICY\_$POLICY2\_mix_perc$PERC.pkl \
        --policy $POLICY \
        --policy2 $POLICY2 \
        --perc $PERC \
        --robosuite
done