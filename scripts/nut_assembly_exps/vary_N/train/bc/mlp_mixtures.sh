#!/bin/bash
ARCH=MLP
DATA_SOURCES=(madeline_sidd_mix_perc0.5 madeline_sidd_mix_perc0.6 madeline_sidd_mix_perc0.7 madeline_sidd_mix_perc0.8 madeline_sidd_mix_perc0.9)
DATE=jan9
EPOCHS=15
ENVIRONMENT=NutAssembly
HIDDEN_SIZES=(128 256 512 1024)
METHOD=BC
NS=(5 10 15 20 25 30)
NUM_MODELS=1
SEEDS=(0 2 4)

if [ $NUM_MODELS -gt 1 ]
then
    EXP_NAME_ARCH=Ensemble$ARCH
else
    EXP_NAME_ARCH=$ARCH
fi

for HIDDEN_SIZE in "${HIDDEN_SIZES[@]}"
do
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
                    --epochs $EPOCHS \
                    --hidden_size $HIDDEN_SIZE \
                    --method $METHOD \
                    --arch $ARCH \
                    --num_models $NUM_MODELS \
                    --seed $SEED \
                    --robosuite
            done
        done
    done
done