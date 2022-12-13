import os
import copy
import pickle
import argparse

import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="./data", help="Directory where data is stored.")
parser.add_argument("--environment", type=str, default="NutAssembly", help="Name of environment for data sampling.")
parser.add_argument("--source", type=str, default="base", help="Name of data file.")

args = parser.parse_args()
source_file = os.path.join(args.data_dir, args.environment, args.source+'.pkl')
dest_file = os.path.join(args.data_dir, args.environment, args.source+'-ld.pkl')
data = pickle.load(open(source_file, "rb"))
new_data = copy.copy(data)

for d, demo in enumerate(data):
    for o, obs in enumerate(demo["obs"]):
        object_state = obs[:29]
        proprio = obs[29:]
        eefpos = proprio[24:27]
        eefquat = proprio[27:31]
        gripper_qpos = proprio[31:37]
        new_obs = torch.cat([eefpos, eefquat, gripper_qpos, object_state])
        new_data[d]["obs"][o] = new_obs
pickle.dump(new_data, open(dest_file, "wb"))