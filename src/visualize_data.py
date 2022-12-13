import argparse
import numpy as np
import os
import pickle
import random
import time

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

import robosuite as suite
from envs import CustomWrapper
from robosuite import load_controller_config
from robosuite.utils.mjcf_utils import postprocess_model_xml
from util.int_utils import get_env, visualize_demo


def parse_args():
    parser = argparse.ArgumentParser()

    # Sampling parameters
    parser.add_argument("--environment", type=str, default="Reach2D", help="Name of environment for data sampling.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for data sampling.")
    parser.add_argument(
        "--robosuite", action="store_true", help="Whether or not the environment is a Robosuite environment"
    )
    parser.add_argument(
        "--N_trajectories", type=int, default=1, help="Number of trajectories (demonstrations) to sample."
    )
    parser.add_argument("--nut_type", type=str, default="round", help="Type of nut to use.")
    parser.add_argument("--use_actions", action="store_true", help="Whether to use actions or states from the data.")

    # Data parameters
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory where data is stored.")
    parser.add_argument("--data_file", type=str, default="data", help="Name of data file.")
    
    # Robosuite
    parser.add_argument("--no_render", action="store_true", help="If provided, Robosuite rendering is skipped.")
    parser.add_argument("--web_interface", action="store_true", help="If provided, a web interface is started.")
    parser.add_argument("--low_dim", action="store_true", help="If provided, low-dimensional state is used.")
    parser.add_argument("--space_mouse", action="store_true", help="If provided, low-dimensional state is used.")

    args = parser.parse_args()
    return args

def main(args):
    random.seed(args.seed)
    env, _ = get_env(args)
    data_file = os.path.join(args.data_dir, args.environment, args.data_file+'.pkl')
    if os.path.exists(data_file):
        with open(data_file, "rb") as f:
            data = pickle.load(f)
    else:
        raise FileNotFoundError(f"Could not find data file {data_file}!")
    random.shuffle(data)
    demos = data[:args.N_trajectories]
    print(f"Sampled {len(demos)} demonstrations.")
    env.reset()
    for demo in demos:
        visualize_demo(demo, env, args)

if __name__ == "__main__":
    args = parse_args()
    main(args)
            