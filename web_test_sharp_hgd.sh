python src/app.py \
        --N  50 \
        --robosuite \
        --exp_name jenn-sharp-hgd-nut-assembly \
        --data_path data/NutAssembly/base-50.pkl \
        --epochs 50 \
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
        --N_eval_trajectories 50 \
        --trajectories_per_rollout  5 \
        --dagger_epochs 1 \
        --likelihood_threshold 0.4 \
        --entropy_threshold 0.15 \
        --show_nearest \
        --sample_threshold 2 \
        --show_bad_samples 3 \
        --window_size 10 \
        --n_procs 4 \
        --online_feedback \
        --batch_size 128 \
        --web_interface \
        --no_render \
        --N_initial 0  \
        --init_dagger_model \
        --model_path ./out/models/nut-base-30.pt \
        --teaching_samples 5