import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np


FIXED_PILLAR_DATA_SOURCES = ["oracle_fixed_pillar_over", "oracle_fixed_pillar_under"]
FIXED_PILLAR_NO_OBS_DATA_SOURCES = ["oracle_fixed_pillar_over_no_obs", "oracle_fixed_pillar_under_no_obs"]
PILLAR_DATA_SOURCES = ["oracle_over", "oracle_under"]
PILLAR_MIX_DATA_SOURCES = ["oracle_mix_over_under"]
PILLAR_MIX_PERCS_DATA_SOURCES = [
    "oracle_mix_perc_over_under_90_10",
    "oracle_mix_perc_over_under_80_20",
    "oracle_mix_perc_over_under_70_30",
    "oracle_mix_perc_over_under_60_40",
]
ORACLE_DATA_SOURCES = ["oracle"]
FIXED_PILLAR_RIGHT_ANGLE_DATA_SOURCES = ["oracle_fixed_pillar_up_right", "oracle_fixed_pillar_right_up"]
FIXED_PILLAR_RIGHT_ANGLE_NO_OBS_DATA_SOURCES = [
    "oracle_fixed_pillar_up_right_no_obs",
    "oracle_fixed_pillar_right_up_no_obs",
]

TRAIN_SEEDS = [0, 2, 4]
NS = [50, 100, 200, 300, 400, 500, 750, 1000]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="LinearModel")
    parser.add_argument("--ckpt_file", type=str, default="model_4.pt")
    parser.add_argument("--data_source", type=str, help="One of [fixed_pillar, pillar]")
    parser.add_argument("--date", type=str, default="dec28")
    parser.add_argument("--environment", type=str, default="Reach2D")
    parser.add_argument("--method", type=str, default="BC")
    parser.add_argument("--num_models", type=int, default=1)
    parser.add_argument("--seed", type=int, default=4)

    args = parser.parse_args()
    return args


def main(args):
    exp_name_arch = args.arch if args.num_models == 1 else "Ensemble" + args.arch
    if args.data_source == "fixed_pillar":
        data_sources = FIXED_PILLAR_DATA_SOURCES
    elif args.data_source == "fixed_pillar_no_obs":
        data_sources = FIXED_PILLAR_NO_OBS_DATA_SOURCES
    elif args.data_source == "fixed_pillar_right_angle":
        data_sources = FIXED_PILLAR_RIGHT_ANGLE_DATA_SOURCES
    elif args.data_source == "fixed_pillar_right_angle_no_obs":
        data_sources = FIXED_PILLAR_RIGHT_ANGLE_NO_OBS_DATA_SOURCES
    elif args.data_source == "pillar":
        data_sources = PILLAR_DATA_SOURCES
    elif args.data_source == "pillar_mix":
        data_sources = PILLAR_MIX_DATA_SOURCES
    elif args.data_source == "pillar_mix_percs":
        data_sources = PILLAR_MIX_PERCS_DATA_SOURCES
    elif args.data_source == "oracle":
        data_sources = ORACLE_DATA_SOURCES
    else:
        raise NotImplementedError
    for data_source in data_sources:
        seed_successes = []
        for N in NS:
            successes = []
            for train_seed in TRAIN_SEEDS:
                exp_name = f"{args.date}/{args.environment}/{args.method}/{exp_name_arch}/eval/{data_source}_N{N}_seed{args.seed}/train_seed{train_seed}"
                exp_dir = os.path.join("./out", exp_name)
                data_file = os.path.join(exp_dir, "eval_auto_data.pkl")

                with open(data_file, "rb") as f:
                    data = pickle.load(f)
                num_successes = sum([traj["success"] for traj in data])
                successes.append(num_successes)
            seed_successes.append(successes)
        means = [np.mean(seed_successes[i]) for i in range(len(seed_successes))]
        stds = [np.std(seed_successes[i]) for i in range(len(seed_successes))]
        plt.errorbar(NS, means, yerr=stds, marker="o", label=f"50% over / 50% under")
    plt.xlabel("N")
    plt.ylabel("# successes")
    plt.ylim(0, 105)
    plt.title(f"{args.method} w/ {exp_name_arch} Auto-Only Rollout")
    plt.legend()
    save_path = (
        f"./out/{args.date}/{args.environment}/{args.method}/{exp_name_arch}/eval/{args.data_source}_successes_vs_N.png"
    )
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
