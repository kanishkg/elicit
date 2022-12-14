python src/app.py \
        --N  30 \
        --robosuite \
        --exp_name ruth-hgd-Nut-5 \
        --init_dagger_model \
        --model_path ./out/models/nut-base-30.pt \
        --data_path data/NutAssembly/base-50.pkl \
        --epochs 1 \
        --policy user \
        --environment NutAssembly \
        --arch MLP \
        --method HGDagger \
        --hidden_size 1024 \
        --num_models 5 \
        --seed 0 \
        --layer_norm \
        --dropout 0.5 \
        --best_epoch \
        --overwrite \
        --N_eval_trajectories 2 \
        --trajectories_per_rollout  5 \
        --dagger_epochs 1 \
        --likelihood_threshold 100000 \
        --entropy_threshold 0. \
        --window_size 10 \
        --n_procs 4 \
        --batch_size 128 \
        --web_interface \
        --no_render \
        --N_initial 0  \
        --teaching_samples 0
