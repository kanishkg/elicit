import os
import h5py
import pickle

import numpy as np
import random
import torch

from datasets.buffer import BufferDataset, BufferSequenceDataset


def get_dataset(data_path, N, save_dir, seq_len=1, images=False):
    train_data, val_data = load_data_from_file(data_path, N, seq_len=seq_len, images=images)
    print(f"saving to: ", os.path.join(save_dir, "train_data.pkl"))
    pickle.dump(train_data, open(os.path.join(save_dir, "train_data.pkl"), "wb"))
    if seq_len == 1:
        train = BufferDataset(train_data)
        val = BufferDataset(val_data)
    elif seq_len > 1:
        train = BufferSequenceDataset(train_data)
        val = BufferSequenceDataset(val_data)
    return train, val

def get_policy_data(file_path, shuffle=False, seq_len=1):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    parsed_demos = [] 
    for demo in data:
        parsed_demos.append(parse_demo([demo], seq_len, shuffle))
    return parsed_demos  

def parse_demo(data, seq_len, shuffle):
    if seq_len == 1:
        # TODO: combine this with general case below  
        obs = []
        act = []
        for demo in data:
            obs.extend(demo["obs"])
            act.extend(demo["act"])
        data = {"obs": torch.stack(obs), "act": torch.stack(act)}
    else:
        assert seq_len > 1 
        # sample sequences of length seq_len from data for training
        obs_seq = []
        act_seq = []
        images_seq = []
        for demo in data:
            num_seq = len(demo["obs"])
            num_seq -= seq_len           
            for i in range(num_seq):
                obs_seq.append(torch.stack(demo["obs"][i : i + seq_len]))
                act_seq.append(torch.stack(demo["act"][i : i + seq_len]))
        data = {"obs": torch.stack(obs_seq).float(), "act": torch.stack(act_seq).float()}

    if shuffle:
        idxs = torch.randperm(len(data["obs"]))
    else:
        idxs = torch.arange(len(data["obs"]))
    data["obs"] = data["obs"][idxs]
    data["act"] = data["act"][idxs]       
    return data

def parse_vis_demo(data, seq_len, shuffle):
    if seq_len == 1:
        # TODO: combine this with general case below  
        obs = []
        act = []
        images = []
        for demo in data:
            obs.extend(demo["obs"])
            act.extend(demo["act"])
        data = {"obs": torch.stack(obs), "act": torch.stack(act), "images": torch.stack(images)}
    else:
        assert seq_len > 1 
        # sample sequences of length seq_len from data for training
        obs_seq = []
        act_seq = []
        images_seq = []
        for demo in data:
            num_seq = len(demo["obs"])
            num_seq -= seq_len           
            for i in range(num_seq):
                obs_seq.append(torch.stack(demo["obs"][i : i + seq_len]))
                act_seq.append(torch.stack(demo["act"][i : i + seq_len]))
        data = {"obs": torch.stack(obs_seq).float(), "act": torch.stack(act_seq).float()}

    if shuffle:
        idxs = torch.randperm(len(data["obs"]))
    else:
        idxs = torch.arange(len(data["obs"]))
    data["obs"] = data["obs"][idxs]
    data["act"] = data["act"][idxs]       
    return data


def load_data_from_file(file_path, N, shuffle=True, seq_len=1, images=False):
    if not images:
        with open(file_path, "rb") as f:
            # data is a list of dictionaries of demonstrations
            # each dictionary has keys: obs, act, states, init_xml
            # each key except xml is a list of torch tensors
            data = pickle.load(f)
            data = np.array(data)
        N_available = len(data)
        if shuffle:
            idxs = torch.randperm(len(data))
            data = data[idxs]
        if N < N_available:

            data = data[:N]

        elif N > N_available:
            print(
                f"Warning: requested size of dataset N={N} is greater than {N_available}, the size of the dataset available."
                " Using entire dataset."
            )
        N = len(data)
        train_demos = data[:int(N * 0.9)]
        val_demos = data[int(N * 0.9) :]
        train_data = parse_demo(train_demos, seq_len, shuffle)
        val_data = parse_demo(val_demos, seq_len, shuffle)
        return train_data, val_data
    
    else:
        data = h5py.File(file_path, "r")["data"]
        N_available = len(data.keys())
        demo_keys = list(data.keys())
        if N < N_available:
            if shuffle:
                idxs = torch.randperm(len(demo_keys))
                demo_keys = demo_keys[idxs]
            demo_keys = demo_keys[:N]

        elif N > N_available:
            print(
                f"Warning: requested size of dataset N={N} is greater than {N_available}, the size of the dataset available."
                " Using entire dataset."
            )
        N = len(demo_keys)
        train_demos = demo_keys[:int(N * 0.9)]
        val_demos = demo_keys[int(N * 0.9) :]
        train_data = parse_vis_demo(train_demos, seq_len, shuffle)
        val_data = parse_vis_demo(val_demos, seq_len, shuffle)
        return train_data, val_data