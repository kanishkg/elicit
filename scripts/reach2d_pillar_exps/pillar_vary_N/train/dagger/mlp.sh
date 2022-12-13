#!/bin/bash
ARCH=MLP
DATA_SOURCES=(oracle_under oracle_over)
DATE=jan4_lenient_hidden150
ENVIRONMENT=Reach2DPillar
EPOCHS=5
HIDDEN_SIZE=150
METHOD=Dagger
NS=(50 100 200 300 400 500 750 1000)
NUM_MODELS=1
SEEDS=(0 2 4)

if [ $NUM_MODELS -gt 1 ]
then
    EXP_NAME_ARCH=Ensemble$ARCH
else
    EXP_NAME_ARCH=$ARCH
fi

for SEED in "${SEEDS[@]}"
do
    for N in "${NS[@]}"
    do
        ((traj_per_rollout=$N/$EPOCHS))
        echo N=$N, traj_per_rollout=$traj_per_rollout
        for DATA_SOURCE in "${DATA_SOURCES[@]}"
        do
            python src/main.py \
                --N $N \
                --trajectories_per_rollout $traj_per_rollout \
                --epochs $EPOCHS \
                --exp_name $DATE/$ENVIRONMENT/$METHOD/$EXP_NAME_ARCH/$DATA_SOURCE\_N$N\_seed$SEED \
                --data_path ./data/$ENVIRONMENT/$DATA_SOURCE.pkl \
                --environment $ENVIRONMENT \
                --hidden_size $HIDDEN_SIZE \
                --method $METHOD \
                --arch $ARCH \
                --num_models $NUM_MODELS \
                --seed $SEED \
                --use_indicator_beta \
                --dagger_beta 1.0
        done
    done
done