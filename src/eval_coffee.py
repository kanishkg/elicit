import argparse
import os
import random
from datetime import datetime

import h5py
import json

import numpy as np
import torch
import wandb
from stable_baselines3.common.vec_env import SubprocVecEnv
import robosuite

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
    COFFEE_MAX_TRAJ_LEN
)
from datasets.util import get_dataset, get_policy_data
from envs import Reach2D, Reach2DPillar
from policies import Reach2DPillarPolicy, Reach2DPolicy, NutAssemblyPolicy
from util import get_model_type_and_kwargs, init_model, setup_robosuite
import time



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
    demo_path = "/iliad/u/kanishkg/IWR_dataset/icra_public/coffee/user_0/more/"
    hdf5_path = os.path.join(demo_path, "demo.hdf5")
    f = h5py.File(hdf5_path, "r")
    hdf5_path2 = os.path.join(demo_path, "states.hdf5")
    f2 = h5py.File(hdf5_path2, "r")
    env_kwargs = json.loads(f2["data"].attrs["env_args"])['env_kwargs']
    env_name = f["data"].attrs["env"]
    update_kwargs = dict(
                has_renderer=False,
                has_offscreen_renderer=False,
                ignore_done=True,
                use_camera_obs=False,
                reward_shaping=True,
                control_freq=40,
    )
    env_kwargs.update(update_kwargs)
    env = robosuite.make(env_name, **env_kwargs)

    # Initialize model
    model_type, model_kwargs = get_model_type_and_kwargs(args, 41, 7)
    model = init_model(model_type, model_kwargs, device=device, num_models=args.num_models)
    model.to(device)
    
    # sample initial trajectories for eval comparison
    demos = sorted(list(f["data"].keys()))
    print('total demos:', len(demos))
    initial_states = []
    for ep in demos:
        init_states = f["data"][ep]["states"]
        initial_states.append(init_states[0])  

    # Get list of model ckpts in out dir
    model_paths = [m for m in os.listdir(save_dir) if m.endswith(".pt") and "best" not in m]
    model_paths = [m for m in model_paths if int(m.split('_')[1][:-3])%args.eval_interval == 0 and int(m.split('_')[1][:-3]) <= args.eval_max]

    def get_ld(observations):
        gripper_qpos = observations["gripper_qpos"]   
        eef_pos = observations["eef_pos"]
        eef_quat = observations["eef_quat"]
        object_states = observations["object-state"]
        state_obs = np.concatenate((gripper_qpos, eef_pos, eef_quat, object_states))
        state_obs = torch.tensor(state_obs).float().to(device)
        state_obs = state_obs.unsqueeze(0)
        return state_obs

    args.no_render = True

    random.shuffle(initial_states)
    for m in model_paths:
        print(f"Evaluating model {m}")
        model_path = os.path.join(save_dir, m)
        epoch_num = int(m.split('_')[1][:-3])
        ckpt = torch.load(model_path, map_location=device)
        if args.num_models > 1:
            for ensemble_model, state_dict in zip(model.models, ckpt["models"]):
                ensemble_model.load_state_dict(state_dict)
        else:
            model.load_state_dict(ckpt["model"])
        model.eval()
        success_count = 0
        for i in range(args.N_eval_trajectories):
            if args.arch == "RNN":
                D = 2 if args.bidirectional else 1
                if args.num_models == 1:
                    hidden_state = (torch.zeros(1, D*args.num_layers, args.hidden_size).to(device),
                                    torch.zeros(1, D*args.num_layers, args.hidden_size).to(device))
                else:
                    hidden_state = (torch.zeros(args.num_models, 1, D*args.num_layers, args.hidden_size).to(device),
                                    torch.zeros(args.num_models, 1,  D*args.num_layers, args.hidden_size).to(device)) 

            assert args.N_eval_trajectories < len(initial_states) 

            init_state = initial_states[i]
            env.sim.set_state_from_flattened(init_state)                
            env.sim.forward()
            observations = env._get_observation()
            curr_obs = get_ld(observations)

            for s in range(600):
                if args.arch == "RNN":
                    # make hidden state contiguous
                    hidden_state = (hidden_state[0].contiguous(), hidden_state[1].contiguous())
                    a, hidden_state = model.get_action(curr_obs, state=hidden_state)
                    a = a.squeeze()
                else:
                    a = model.get_action(curr_obs).to(device)
                next_obs, success, done, _ = env.step(a.detach().cpu().numpy())
                if env._check_success()["task"]:
                    success_count += 1
                    break
                next_obs = get_ld(next_obs)
                curr_obs = next_obs

            wandb.run.summary["success_rate"] = success_count /  (i+1)      
        
        # log rewards with epoch num in wandb
        wandb.log(
                {"epoch": epoch_num, "success_rate": success_count / args.N_eval_trajectories}
            ) 

if __name__ == "__main__":
    args = parse_args()
    main(args)