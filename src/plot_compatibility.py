import os
import random
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt

from constants import (
    NUT_ASSEMBLY_MAX_TRAJ_LEN,
    PICKPLACE_MAX_TRAJ_LEN,
)

from datasets.util import get_policy_data
from util.int_utils import load_model, get_env, get_demo_compatibility, count_bad_samples, count_good_samples


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="NutAssembly", help="Name of environment for data sampling.")
    parser.add_argument("--robosuite", action="store_true", help="Whether to use robosuite environment.")
    parser.add_argument("--hidden_size", type=int, default=512, help="Hidden size of model.")
    parser.add_argument("--model_path", type=str, default="", help="Path to model checkpoint.")
    parser.add_argument("--out_dir", type=str, default="./plots", help="Path to save plots.")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to data directory.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--num_models", type=int, default=1, help="Number of models to evaluate.")
    parser.add_argument("--arch", type=str, default="MLP", help="Architecture of model.") 
    parser.add_argument("--policies", type=str, nargs="+", default=[], help="Policies to evaluate.")
    parser.add_argument("--no_render", action="store_true", help="Whether to render policy.")
    parser.add_argument("--colors", nargs="+", default=[], help="Color of scatter plot.")
    parser.add_argument("--nut_type", type=str, default="round", help="Nut type.")
    parser.add_argument("--low_dim", action="store_true", help="whether to use low-dimensional policy.")
    parser.add_argument("--seq_len", type=int, default=1, help="Sequence length.")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers.")
    parser.add_argument("--bidirectional", action="store_true", help="Whether to use bidirectional LSTM.")
    parser.add_argument("--likelihood_threshold", type=float, default=0.5, help="Threshold for likelihood.")
    parser.add_argument("--entropy_threshold", type=float, default=0.025, help="Threshold for entropy.")

    parser.add_argument("--layer_norm", action="store_true", help="Whether to use low-dimensional policy.")

    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate.")
    parser.add_argument("--normalize", action="store_true", help="Whether to normalize policy.")
    parser.add_argument("--backward", action="store_true", help="backward compatibility.")
    parser.add_argument("--N", type=int, default=1, help="Number of demos to sample.")
    return parser.parse_args()


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    args.web_interface = False
    args.space_mouse = False
    if not os.path.isdir(args.out_dir):
        raise FileExistsError(f"{args.out_dir} does not exist!")
    # Set up environment
    env, _ = get_env(args)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    action_lim = env.action_space.high[0]
    # get policy data to evaluate compatibility
    # RNNs are used as open-loop models
    policy_files = [os.path.join(args.data_dir, args.environment, f+'.pkl') for f in args.policies]
    policies = [get_policy_data(policy) for policy in policy_files]
    # Initialize model
    model = load_model(args, obs_dim, act_dim, device)

    # match colors to policy names
    color_policy = {policy: '#'+color for policy, color in zip(args.policies, args.colors)}
    model.eval()
    compatibility_dict = {}
    for policy, policy_name in zip(policies, args.policies):
        policy = policy[:args.N]
        num_bad = []
        num_good = []
        for i, demo in enumerate(policy):
            comp_matrix = get_demo_compatibility(model, demo, device, action_lim=action_lim)
            bad = count_bad_samples(comp_matrix, args)
            good = count_good_samples(comp_matrix, args)
            num_bad.append(bad)
            num_good.append(good)
        # plot histogram and save fig using list num_bad
        num_good = [x for _, x in sorted(zip(num_bad, num_good), key=lambda pair: pair[0])]
        # fig, ax = plt.subplots()
        # ax.hist(num_bad, bins=[0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 100, 200])
        # ax.plot(num_good)
        # ax.set_title(f"{policy_name}")
        # ax.set_ylabel("Number of good samples")
        # ax.set_xlabel("demo #")
        # fig.savefig(os.path.join(args.out_dir, f"{policy_name}.png"))
        demos = {}
        demos['obs'] = torch.cat([d['obs'] for d in policy])
        demos['act'] = torch.cat([d['act'] for d in policy])
        print(f"Evaluating compatibility for {policy_name}")
        compatibility_dict[f"{policy_name}"] = get_demo_compatibility(model, demos, device, action_lim=action_lim)
        matrix = compatibility_dict[f"{policy_name}"]
        fig, ax = plt.subplots()
        num_bad = count_bad_samples(matrix, args)
        num_good = count_good_samples(matrix, args)
        print(f"{policy_name}: {num_bad} bad samples, {num_good} good samples, {matrix.shape} total samples")

        ax.scatter(matrix[:, 0], matrix[:, 1], s=10, label="", c=color_policy[policy_name],
                edgecolors='none')
        if args.arch == "MLP":
            ax.set_xlim([0.0, 1.])
            ax.set_ylim([0.0, .2])
        elif args.arch == "RNN":
            ax.set_xlim([0.0, 1.])
            ax.set_ylim([0.0, .5])
        elif args.arch == "MDN":
            ax.set_ylim([1.014**0.5, 1.023**0.5])
            ax.set_xlim([-11, -5])

        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_title(f"")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='x', which='both', length=0)
        ax.tick_params(axis='y', which='both', length=0)
        fig.savefig(os.path.join(args.out_dir, f"{policy_name}_{args.arch}_{args.environment}_compatibility.png"))
 

if __name__ == "__main__":
    args = parse_args()
    main(args)

