#!/bin/bash
ARCH=MLP
CHECKPOINT_FILE=model_best.pt
DATA_SOURCES=(oracle_mix_perc_over_under_90_10 oracle_mix_perc_over_under_80_20 oracle_mix_perc_over_under_70_30 oracle_mix_perc_over_under_60_40)
DATE=jan5_lenient_hidden150
ENVIRONMENT=Reach2DPillar
HIDDEN_SIZE=150
METHOD=BC
NS=(50 100 200 300 400 500 750 1000)
NUM_MODELS=1
EVAL_SEED=4
TRAIN_SEEDS=(0 2 4)

if [ $NUM_MODELS -gt 1 ]
then
    EXP_NAME_ARCH=Ensemble$ARCH
else
    EXP_NAME_ARCH=$ARCH
fi

for TRAIN_SEED in "${TRAIN_SEEDS[@]}"
do
    for N in "${NS[@]}"
    do
        for DATA_SOURCE in "${DATA_SOURCES[@]}"
        do
            python src/main.py \
                --N $N \
                --eval_only \
                --model_path ./out/$DATE/$ENVIRONMENT/$METHOD/$EXP_NAME_ARCH/$DATA_SOURCE\_N$N\_seed$TRAIN_SEED/$CHECKPOINT_FILE \
                --exp_name $DATE\_less_lenient/$ENVIRONMENT/$METHOD/$EXP_NAME_ARCH/eval/$DATA_SOURCE\_N$N\_seed$EVAL_SEED/train_seed$TRAIN_SEED \
                --data_path ./data/$ENVIRONMENT/$DATA_SOURCE.pkl \
                --environment $ENVIRONMENT \
                --method $METHOD \
                --arch $ARCH \
                --hidden_size $HIDDEN_SIZE \
                --num_models $NUM_MODELS \
                --seed $EVAL_SEED 
        done
    done
done
