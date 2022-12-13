NAME=colton
ENVIRONMENT=HammerPlaceEnv
NPRACTICE=0
NTEACH=5
NLEARN=5
LIK=0.4
ENT=0.1
BADLIM=3
SHOWBAD=3

python src/app.py \
        --N  50 \
        --robosuite \
        --exp_name $NAME-sharp-dagger-Hammer-1-5-$LIK-$ENT \
        --data_path data/$ENVIRONMENT/hgtinynbase-20.pkl \
        --epochs 50 \
        --policy user \
        --environment $ENVIRONMENT \
        --arch MLP \
        --method Dagger \
        --hidden_size 1024 \
        --num_models 5 \
        --seed 0 \
        --layer_norm \
        --dropout 0.5 \
        --best_epoch \
        --overwrite \
        --N_eval_trajectories 50 \
        --trajectories_per_rollout $NTEACH \
        --dagger_epochs 1 \
        --likelihood_threshold $LIK \
        --entropy_threshold $ENT \
        --show_nearest \
        --sample_threshold $BADLIM \
        --show_bad_samples $SHOWBAD \
        --window_size 10 \
        --n_procs 4 \
        --online_feedback \
        --batch_size 128 \
        --web_interface \
        --no_render \
        --N_initial $NPRACTICE \
        --init_dagger_model \
        --model_path ./out/models/hammer_web.pt \
        --teaching_samples $NLEARN
