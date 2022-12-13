NAME=colton
ENVIRONMENT=HammerPlaceEnv
NPRACTICE=2
NTEACH=5

python src/app.py \
        --N 50 \
        --robosuite \
        --exp_name $NAME-dagger-hammer-$NTEACH \
        --data_path data/$ENVIRONMENT/hgtinynbase-20.pkl \
        --epochs 1 \
        --policy user \
        --environment $ENVIRONMENT \
        --arch MLP \
        --method Dagger \
        --hidden_size 512 \
        --num_models 5 \
        --seed 0 \
        --layer_norm \
        --dropout 0.5 \
        --best_epoch \
        --overwrite \
        --N_eval_trajectories 50 \
        --trajectories_per_rollout $NTEACH \
        --dagger_epochs 1 \
        --likelihood_threshold 100000 \
        --entropy_threshold 0. \
        --window_size 10 \
        --n_procs 4 \
        --web_interface \
        --batch_size 128 \
        --no_render \
        --N_initial $NPRACTICE  \
        --teaching_samples 0
