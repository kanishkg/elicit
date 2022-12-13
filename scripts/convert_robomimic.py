import os
import pickle
import argparse

import h5py
import numpy as np
import torch


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="./data", help="Directory where data is stored.")
parser.add_argument("--environment", type=str, default="NutAssembly", help="Name of environment for data sampling.")
parser.add_argument("--source", type=str, default="low_dim.hdf5", help="Name of data file.")
parser.add_argument("--filter", type=str, default=None, help="Filter for demos.")
parser.add_argument("--dest", type=str, default="data.pkl", help="Name of pkl file.")
parser.add_argument("--low_dim", action="store_true", help="Whether to use low-dim data.")

low_dim_keys = [
            "robot0_eef_pos", 
            "robot0_eef_quat", 
            "robot0_gripper_qpos", 
            "object",
        ]
keys_to_include = ['robot0_joint_pos_cos', 'robot0_joint_pos_sin',
                  'robot0_joint_vel', 'robot0_eef_pos', 'robot0_eef_quat',
                  'robot0_gripper_qpos', 'robot0_gripper_qvel', 'object']

low_dim_keys_two_arm = [
            "robot0_eef_pos", 
            "robot0_eef_quat", 
            "robot0_gripper_qpos", 
            "robot1_eef_pos", 
            "robot1_eef_quat", 
            "robot1_gripper_qpos", 
            "object",
        ]

def get_demos(data, demo_list, keys):
    demos = []
    for b in demo_list:
        obs = np.concatenate([data[b]['obs'][k][()] for k in keys], axis=1).tolist()
        obs = [torch.tensor(o) for o in obs]
        act = [torch.tensor(a) for a in np.array(data[b]['actions'][()]).tolist()]
        states = [torch.tensor(s) for s in np.array(data[b]['states'][()]).tolist()]
        demos.append({"obs": obs, "act":act, "states":states, "init_xml": data[b].attrs["model_file"]})
    return demos
args = parser.parse_args()
source_file = os.path.join(args.data_dir, args.environment, args.source)
dest_file = os.path.join(args.data_dir, args.environment, args.dest)
f = h5py.File(source_file,'r+')
if args.filter:
    demo_list = f['mask'][args.filter][()].tolist()
else:
    demo_list = list(f['data'].keys())
if args.low_dim:
    if "TwoArm" in args.environment:
        keys = low_dim_keys_two_arm
    else:
        keys = low_dim_keys
else:
    keys = keys_to_include
print(f"Found {len(demo_list)} demonstrations.")
print("Using keys:", keys)
demos = get_demos(f['data'], demo_list, keys)
pickle.dump(demos, open(dest_file,"wb"))