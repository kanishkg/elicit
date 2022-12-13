import argparse
import os
import random
from datetime import datetime


import numpy as np
import torch
import wandb
from stable_baselines3.common.vec_env import SubprocVecEnv

from algos import BC, Dagger, HGDagger
from main import get_policy
from constants import (
    MAX_NUM_LABELS,
    NUT_ASSEMBLY_MAX_TRAJ_LEN,
    NUT_ASSEMBLY_SQUARE_MAX_TRAJ_LEN,
    DOOR_MAX_TRAJ_LEN,
    WIPE_MAX_TRAJ_LEN,
    PICKPLACE_MAX_TRAJ_LEN,
    REACH2D_MAX_TRAJ_LEN,
    REACH2D_PILLAR_MAX_TRAJ_LEN,
    REACH2D_RANGE_X,
    REACH2D_RANGE_Y,
    CAN_MAX_TRAJ_LEN,
    TRANSPORT_MAX_TRAJ_LEN,
    COFFEE_MAX_TRAJ_LEN,
    HAMMER_MAX_TRAJ_LEN
)
from datasets.util import get_dataset, get_policy_data
from envs import Reach2D, Reach2DPillar
from policies import Reach2DPillarPolicy, Reach2DPolicy, NutAssemblyPolicy
from util import get_model_type_and_kwargs, init_model, setup_robosuite



def parse_args():
    parser = argparse.ArgumentParser()

    # Logging + output
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Unique experiment ID for saving/logging purposes. If not provided, date/time is used as default.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./out",
        help="Parent output directory. Files will be saved at /\{args.out_dir\}/\{args.exp_name\}.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If provided, the save directory will be overwritten even if it exists already.",
    )
    parser.add_argument("--save_iter", type=int, default=5, help="Checkpoint will be saved every args.save_iter epochs.")

    # Data loading
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/dec14_gen_oracle_reach2d_data_1k/dec14_gen_oracle_reach2d_data_1k_s4/pick-place-data-1000.pkl",
    )
    parser.add_argument("--N", type=int, default=1000, help="Size of dataset.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--low_dim", action="store_true", help="Use low-dimensional dataset.")
    # Autonomous evaluation only
    parser.add_argument(
        "--eval_only", action="store_true", help="If true, rolls out the autonomous policy of the provided trained model"
    )
    parser.add_argument(
        "--model_path", type=str, default=None, help="Path to saved model checkpoint for evaulation purposes."
    )
    parser.add_argument(
        "--N_eval_trajectories",
        type=int,
        default=100,
        help="Number of trajectories to roll out for autonomous-only evaluation.",
    )
    parser.add_argument(
        "--n_procs", type=int, default=1, help="Number of processes to use for evaluation.")

    # Environment details + rendering
    parser.add_argument("--environment", type=str, default="Reach2D", help="Environment name")
    parser.add_argument(
        "--robosuite", action="store_true", help="Whether or not the environment is a Robosuite environment"
    )

    parser.add_argument("--no_render", action="store_true", help="If true, Robosuite rendering is skipped.")
    parser.add_argument("--random_start_state", action="store_true", help="Random start state for Reach2D environment")
    parser.add_argument("--nut_type", type=str, default="round", help="Nut type for environment")
    parser.add_argument("--use_actions", action="store_true", help="If provided, actions are used in vusualizing the policy.")
    parser.add_argument("--web_interface", action="store_true", help="If provided, a web interface is started.")
    parser.add_argument("--space_mouse", action="store_true", help="If provided, space mouse is used for controlling.")


    # Method / Model details
    parser.add_argument(
        "--method", type=str, default="BC", help="One of \{BC, Dagger, ThriftyDagger, HGDagger, LazyDagger\}}"
    )
    parser.add_argument("--arch", type=str, default="LinearModel", help="Model architecture to use.")
    parser.add_argument("--hidden_size", type=int, default=32, help="Hidden size of MLP if args.arch == 'MLP'")
    parser.add_argument(
        "--num_models", type=int, default=1, help="Number of models in the ensemble; if 1, a non-ensemble model is used"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--layer_norm", action="store_true", help="Whether or not to use layer normalization.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate.")
    parser.add_argument("--num_layers", type=int, default=2, help="number of layers in rnn.")
    parser.add_argument("--bidirectional", action="store_true", help="Whether or not to use bidirectional rnn.")
    parser.add_argument("--seq_len", type=int, default=1, help="Sequence length.")
    # Dagger-specific parameters
    parser.add_argument(
        "--dagger_beta",
        type=float,
        default=0.9,
        help="DAgger parameter; policy will be (beta * expert_action) + (1-beta) * learned_policy_action",
    )
    parser.add_argument(
        "--use_indicator_beta",
        action="store_true",
        help="DAgger parameter; policy will use beta=1 for first iteration and beta=0 for following iterations.",
    )
    parser.add_argument(
        "--dagger_epochs", type=int, default=1, help="Number of expert policy interactions."
    )
    parser.add_argument(
        "--check_compatibility", action="store_true", help="Check compatibility of expert and model policies."
    )
 
    # Training parameters
    parser.add_argument("--epochs", type=int, default=5, help="Number of iterations to run overall method for")
    parser.add_argument("--clip", type=float, default=10, help="Value to clip gradients by")


    parser.add_argument("--policy", type=str, default="over", help="Specifies which expert policy to use for interactions.")
    parser.add_argument("--offline_policy", action="store_true", help="If true, uses offline policy dataset for interactions.")
    parser.add_argument("--likelihood_filter", type=float, default=1e6, help="Likelihood filter for offline policy dataset.")
    parser.add_argument("--entropy_filter", type=float, default=0.0, help="Entropy filter for offline policy dataset.")
    parser.add_argument("--normalize", action="store_true", help="Normalize observations.")

    parser.add_argument(
        "--trajectories_per_rollout",
        type=int,
        default=10,
        help=(
            "Number of trajectories to roll out per epoch, required for interactive methods and ignored for offline data"
            " methods."
        ),
    )
    parser.add_argument("--best_epoch", action="store_true", help="whether to use best epoch of the model for evaluation or the last epoch")

    # Random seed
    parser.add_argument("--seed", type=int, default=0)

    # Feedback args
    parser.add_argument("--sampling_method", type=str, default="likelihood", help="Sampling method initial demonstration.")
    parser.add_argument("--teaching_samples", type=int, default=0, help="Number of teaching samples.")
    parser.add_argument("--likelihood_threshold", type=float, default=0., help="Threshold for likelihood.")
    parser.add_argument("--entropy_threshold", type=float, default=1e6, help="Threshold for entropy.")
    parser.add_argument("--show_nearest", action="store_true", help="Whether to show the nearest demonstration.")
    parser.add_argument("--filter_method", type=str, default="sample", help="Filter method for initial demonstrations.")
    parser.add_argument("--sample_threshold", type=int, default=0, help="Threshold for sample filter.")
    parser.add_argument("--show_bad_samples", type=int, default=0, help="Number of bad samples to show for feedback.")
    parser.add_argument("--window_size", type=int, default=5, help="Window size for showing trajectory around a bad sample.")
    parser.add_argument("--online_feedback", action="store_true", help="Whether to use online feedback.")

    # Evaluation args
    parser.add_argument("--eval_interval", type=int, default=50, help="Number of epochs between evaluations.")
    parser.add_argument("--eval_max", type=int, default=2000, help="Max number of epoch to evaluate for.")
    return parser.parse_args()

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Set up output directories
    if args.exp_name is None:
        args.exp_name = datetime.now().strftime("%m-%d-%Y-%H:%M:%S")
    wandb_name = f"eval-{args.environment}-{args.method}-{args.arch}-{args.num_models}-{args.hidden_size}"
    wandb.init(id=wandb.util.generate_id(),
            name=wandb_name,
            config=args,
            entity='kanishkgandhi',
            project='sharp-dagger',
            reinit=True)

    save_dir = os.path.join(args.out_dir, args.exp_name)
    if not args.overwrite and os.path.isdir(save_dir):
        raise FileExistsError(
            f"The directory {save_dir} already exists. If you want to overwrite it, rerun with the argument --overwrite."
        )
    os.makedirs(save_dir, exist_ok=True)

    # Set up environment
    if args.robosuite:
        if args.environment == "PickPlace":
            env, robosuite_cfg = setup_robosuite(args, max_traj_len=PICKPLACE_MAX_TRAJ_LEN)
            obs_dim = env.observation_space.shape[0]
            act_dim = env.action_space.shape[0]
        elif args.environment == "NutAssembly":
            if args.nut_type == "round":
                env, robosuite_cfg = setup_robosuite(args, max_traj_len=NUT_ASSEMBLY_MAX_TRAJ_LEN)
            elif args.nut_type == "square":
                env, robosuite_cfg = setup_robosuite(args, max_traj_len=NUT_ASSEMBLY_SQUARE_MAX_TRAJ_LEN)
            obs_dim = env.observation_space.shape[0]
            act_dim = env.action_space.shape[0]
        elif args.environment == "Wipe":
            env, robosuite_cfg = setup_robosuite(args, max_traj_len=WIPE_MAX_TRAJ_LEN)
            obs_dim = env.observation_space.shape[0]
            act_dim = env.action_space.shape[0]
        elif args.environment == "HammerPlaceEnv":
            env, robosuite_cfg = setup_robosuite(args, max_traj_len=HAMMER_MAX_TRAJ_LEN)
            obs_dim = env.observation_space.shape[0]
            act_dim = env.action_space.shape[0]
        elif args.environment == "SawyerCoffee":
            env, robosuite_cfg = setup_robosuite(args, max_traj_len=COFFEE_MAX_TRAJ_LEN)
            obs_dim = env.observation_space.shape[0]
            act_dim = env.action_space.shape[0]
        elif args.environment == "TwoArmTransport":
            env, robosuite_cfg = setup_robosuite(args, max_traj_len=TRANSPORT_MAX_TRAJ_LEN)
            obs_dim = env.observation_space.shape[0]
            act_dim = env.action_space.shape[0]
        elif args.environment == "Door":
            env, robosuite_cfg = setup_robosuite(args, max_traj_len=DOOR_MAX_TRAJ_LEN)
            obs_dim = env.observation_space.shape[0]
            act_dim = env.action_space.shape[0]
        elif args.environment == "PickPlaceCan":
            env, robosuite_cfg = setup_robosuite(args, max_traj_len=CAN_MAX_TRAJ_LEN)
            obs_dim = env.observation_space.shape[0]
            act_dim = env.action_space.shape[0]
        else:
            raise NotImplementedError(f"Environment {args.environment} has not been implemented yet!")
    elif args.environment == "Reach2D":
        env = Reach2D(
            device,
            max_ep_len=REACH2D_MAX_TRAJ_LEN,
            random_start_state=args.random_start_state,
            range_x=REACH2D_RANGE_X,
            range_y=REACH2D_RANGE_Y,
        )
        robosuite_cfg = None
        obs_dim = env.obs_dim
        act_dim = env.act_dim
    elif args.environment == "Reach2DPillar":
        env = Reach2DPillar(
            device,
            max_ep_len=REACH2D_PILLAR_MAX_TRAJ_LEN,
            random_start_state=args.random_start_state,
            range_x=REACH2D_RANGE_X,
            range_y=REACH2D_RANGE_Y,
        )
        robosuite_cfg = None
        obs_dim = env.obs_dim
        act_dim = env.act_dim
    else:
        raise NotImplementedError(f"Environment {args.environment} has not been implemented yet!")

    # Initialize model
    model_type, model_kwargs = get_model_type_and_kwargs(args, obs_dim, act_dim)
    model = init_model(model_type, model_kwargs, device=device, num_models=args.num_models)
    model.to(device)
    

    if args.eval_only:
        model.eval()
        ckpt = torch.load(args.model_path, map_location=device)
        if args.num_models > 1:
            for ensemble_model, state_dict in zip(model.models, ckpt["models"]):
                ensemble_model.load_state_dict(state_dict)
        else:
            model.load_state_dict(ckpt["model"])

    # Set up method
    if args.method == "Dagger":
        expert_policy = get_policy(args, env, robosuite_cfg)
        algorithm = Dagger(
            model,
            model_kwargs,
            expert_policy=expert_policy,
            device=device,
            save_dir=save_dir,
            beta=args.dagger_beta,
            use_indicator_beta=args.use_indicator_beta,
            max_num_labels=MAX_NUM_LABELS,
            policy_cls=args.arch,
            lr=args.lr
        )
    elif args.method == "HGDagger":
        expert_policy = get_policy(args, env, robosuite_cfg)
        algorithm = HGDagger(model, model_kwargs, expert_policy=expert_policy, device=device, save_dir=save_dir, lr=args.lr)
    elif args.method == "BC":
        algorithm = BC(model, model_kwargs, device=device, save_dir=save_dir, policy_cls=args.arch, lr=args.lr)
    else:
        raise NotImplementedError(f"Method {args.method} has not been implemented yet!")



    ckpt = torch.load(args.model_path, map_location=device)
    if args.num_models > 1:
        for ensemble_model, state_dict in zip(algorithm.model.models, ckpt["models"]):
            ensemble_model.load_state_dict(state_dict)
    else:
        algorithm.model.load_state_dict(ckpt["model"])
    algorithm.model.eval()
    env.env.seed(args.seed)
    algorithm.eval_auto(args, env=env, robosuite_cfg=robosuite_cfg)

if __name__ == "__main__":
    args = parse_args()
    main(args)