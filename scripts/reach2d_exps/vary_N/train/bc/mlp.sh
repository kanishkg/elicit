#!/bin/bash
ARCH=MLP
DATA_SOURCES=(oracle)
DATE=jan3_drop_layer_hidden20
ENVIRONMENT=Reach2D
HIDDEN_SIZE=20
METHOD=BC
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
        for DATA_SOURCE in "${DATA_SOURCES[@]}"
        do
            python src/main.py \
                --N $N \
                --exp_name $DATE/$ENVIRONMENT/$METHOD/$EXP_NAME_ARCH/$DATA_SOURCE\_N$N\_seed$SEED \
                --data_path ./data/$ENVIRONMENT/$DATA_SOURCE.pkl \
                --environment $ENVIRONMENT \
                --hidden_size $HIDDEN_SIZE \
                --method $METHOD \
                --arch $ARCH \
                --num_models $NUM_MODELS \
                --seed $SEED 
        done
    done
done
