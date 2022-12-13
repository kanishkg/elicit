import argparse
import numpy as np
import os
import pickle
import random

import torch

from constants import (
    NUT_ASSEMBLY_MAX_TRAJ_LEN,
    PICKPLACE_MAX_TRAJ_LEN,
    REACH2D_ACT_MAGNITUDE,
    REACH2D_MAX_TRAJ_LEN,
    REACH2D_PILLAR_MAX_TRAJ_LEN,
    REACH2D_RANGE_X,
    REACH2D_RANGE_Y,
    ACTION_BATCH_SIZE
)
from policies import NutAssemblyPolicy
from util.int_utils import get_env, visualize_demo, load_model, get_demo_compatibility, count_bad_samples, sample_trajectory, show_bad_samples, get_nearest
from datasets.util import get_policy_data


def parse_args():
    parser = argparse.ArgumentParser()

    # Sampling parameters
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--environment", type=str, default="NutAssembly", help="Name of environment for data sampling.")
    parser.add_argument(
        "--robosuite", action="store_true", help="Whether or not the environment is a Robosuite environment"
    )
    parser.add_argument(
        "--N_trajectories", type=int, default=1000, help="Number of trajectories (demonstrations) to sample."
    )

    # Saving
    parser.add_argument("--save_dir", default="./data", type=str, help="Directory to save the data in.")
    parser.add_argument("--save_fname", type=str, help="File name for the saved data.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If true and the save file exists already, it will be overwritten with newly-generated data.",
    )

    parser.add_argument("--policy", type=str, default="user", help="Specifies which policy to use.")
    parser.add_argument("--train_policy", type=str, default="kanishk-100", help="Specifies the data the policy was trained on.")

    parser.add_argument("--data_dir", default="./data", type=str, help="Directory to save the data in.")

    parser.add_argument(
        "--model_path", type=str, default=None, help="Model path for the trained policy."
    )
    parser.add_argument("--arch", type=str, default="MLP", help="Model architecture to use.")
    parser.add_argument(
        "--num_models", type=int, default=1, help="Number of models in the ensemble; if 1, a non-ensemble model is used"
    )
    parser.add_argument("--low_dim", action="store_true", help="Whether to use low-dimensional policy.")
    parser.add_argument("--layer_norm", action="store_true", help="Whether to use low-dimensional policy.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate.")
    parser.add_argument("--normalize", action="store_true", help="Whether to normalize policy.")
    parser.add_argument("--hidden_size", type=int, default=512, help="Hidden size of the policy.")
    parser.add_argument("--backward", action="store_true", help="Backward comp for old models.")

    # NutAssembly specific arguments
    parser.add_argument("--nut_type", type=str, default="round", help="Type of nut to use.")

    # Robosuite
    parser.add_argument("--no_render", action="store_true", help="If provided, Robosuite rendering is skipped.")
    parser.add_argument("--use_actions", action="store_true", help="If provided, actions are used in the policy.")

    # Feedback args
    parser.add_argument("--sampling_method", type=str, default="likelihood", help="Sampling method initial demonstration.")
    parser.add_argument("--teaching_samples", type=int, default=3, help="Number of teaching samples.")
    parser.add_argument("--likelihood_threshold", type=float, default=0.5, help="Threshold for likelihood.")
    parser.add_argument("--entropy_threshold", type=float, default=0.0, help="Threshold for entropy.")
    parser.add_argument("--show_nearest", action="store_true", help="Whether to show the nearest demonstration.")
    parser.add_argument("--filter_method", type=str, default="sample", help="Filter method for initial demonstrations.")
    parser.add_argument("--sample_threshold", type=int, default=5, help="Threshold for sample filter.")
    parser.add_argument("--show_bad_samples", type=int, default=0, help="Threshold for sample filter.")
    parser.add_argument("--window_size", type=int, default=5, help="Window size for showing trajectory around a bad sample.")

    args = parser.parse_args()
    return args

def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    save_path = os.path.join(args.save_dir, args.save_fname)
    if not args.overwrite and os.path.isfile(save_path):
        raise FileExistsError(
            f"The file {save_path} already exists. If you want to overwrite it, rerun with the argument --overwrite."
        )
    os.makedirs(args.save_dir, exist_ok=True)

    env, robosuite_cfg = get_env(args)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    action_lim = env.action_space.high[0]
    print(f"obs_dim: {obs_dim}, act_dim: {act_dim}, action_lim: {action_lim}")
    # load model
    model = load_model(args, obs_dim, act_dim, device)    
    model.eval()
    # get interactive policy
    policy_file = os.path.join(args.data_dir, args.environment, args.train_policy+'.pkl')
    policy_demos = get_policy_data(policy_file) 

    # compute compatibility of training set with model
    train_compatibility = [get_demo_compatibility(model, demo, device, action_lim=action_lim) for demo in policy_demos]
    train_likelihood = [np.mean(t[:, 0]) for t in train_compatibility]
    train_entropy = [np.mean(t[:, 1]) for t in train_compatibility]

    if args.sampling_method == 'random':
        # shuffle the demos, likelihood and entropy simultaneously
        zip_demos = list(zip(policy_demos, train_likelihood, train_entropy))
        random.shuffle(zip_demos)
        policy_demos, train_likelihood, train_entropy = list(zip(*zip_demos))
    elif args.sampling_method == 'likelihood':
        # sort policy demos by train likelihood
        policy_demos = [x for _, x in sorted(zip(train_likelihood, policy_demos), key=lambda pair: pair[0])]
        train_entropy = [x for _, x in sorted(zip(train_likelihood, train_entropy), key=lambda pair: pair[0])]
        train_likelihood = sorted(train_likelihood)

    print(f"Train likelihood: {np.mean(train_likelihood)} Train entropy: {np.mean(train_entropy)}")
    num_bad = [count_bad_samples(t, args) for t in train_compatibility]
    print(f"Number of bad samples: {np.mean(num_bad)}")

    ratios =[num_bad[i]/ train_compatibility[i].shape[0] for i in range(len(policy_demos))] 
    print(np.mean(ratios))
    print(f"top 10 demos:{[(num_bad[i], train_compatibility[i].shape[0])for i in range(10)]}")
    print(f"last 10 demos:{[(num_bad[-i], train_compatibility[-i].shape[0])for i in range(10)]}")

    for n in range(args.teaching_samples):
        visualize_demo(policy_demos[n], env, args)

    # create interactive policy
    policy = NutAssemblyPolicy("user", env, robosuite_cfg)

    demos = []
    while len(demos) < args.N_trajectories:

        try:
            demo = sample_trajectory(env, policy)
        except KeyboardInterrupt:
            print("Interrupted, saving demos")
            break

        # obs x 2 (likelihood, entropy)
        comp_matrix = get_demo_compatibility(model, demo, device, action_lim=action_lim)
        demo_likelihood = np.mean(comp_matrix[:,0])
        demo_entropy = np.mean(comp_matrix[:,1])

        print(f"Demo likelihood: {demo_likelihood} Demo entropy: {demo_entropy}")
        print(f"Train likelihood: {np.mean(train_likelihood)} Train entropy: {np.mean(train_entropy)}")
        
        num_bad = count_bad_samples(comp_matrix, args)
        if args.show_bad_samples>0:
            show_bad_samples(comp_matrix, demo, env, args)
        print(f"Number of bad samples: {num_bad}/{comp_matrix.shape[0]}")

        rejected = False
        if args.filter_method == 'aggregate':
            if demo_likelihood > args.likelihood_threshold and demo_entropy < args.entropy_threshold:
                demos.append(demo)
                print(f"Demo accepted: {len(demos)}/{args.N_trajectories} added")
        elif args.filter_method == 'sample':
            if num_bad > args.sample_threshold:
                rejected = True
        
        if rejected:
            print(f"Demo rejected: {len(demos)}/{args.N_trajectories}")
            if args.show_nearest:
                nearest_training_demo = get_nearest(demo, policy_demos) 
                visualize_demo(nearest_training_demo, env, args)

    if len(demos) > 0:
        print("Data generated! Saving data...")
        with open(save_path, "wb") as f:
            pickle.dump(demos, f)
        print(f"{len(demos)} demos saved to {save_path}!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
