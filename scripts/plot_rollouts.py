import argparse
import os
import pickle

import matplotlib.pyplot as plt
import torch


NS = [50, 100, 200, 300, 400, 500, 750, 1000]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="MLP")
    parser.add_argument("--ckpt_file", type=str, default="model_best.pt")
    parser.add_argument("--data_source", type=str, default="oracle_under")
    parser.add_argument("--date", type=str, default="jan4_lenient_hidden150_rerun_less_lenient")
    parser.add_argument("--environment", type=str, default="Reach2DPillar")
    parser.add_argument("--method", type=str, default="BC")
    parser.add_argument("--num_rollouts", type=int, default=20)
    parser.add_argument("--num_models", type=int, default=1)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--train_seed", type=int, default=2)

    args = parser.parse_args()
    return args


def main(args):
    exp_name_arch = args.arch if args.num_models == 1 else "Ensemble" + args.arch
    for N in NS:
        exp_name = f"{args.date}/{args.environment}/{args.method}/{exp_name_arch}/eval/{args.data_source}_N{N}_seed{args.seed}/train_seed{args.train_seed}"
        exp_dir = os.path.join("./out", exp_name)
        data_file = os.path.join(exp_dir, "eval_auto_data.pkl")

        with open(data_file, "rb") as f:
            data = pickle.load(f)
        for i, demo in enumerate(data):
            if i == args.num_rollouts:
                break
            state_xs = []
            state_ys = []
            goal = demo["obs"][0][2:4].detach()
            pillar = demo["obs"][0][4:].detach()
            success = "success" if demo["success"] else "fail"
            for obs in demo["obs"]:
                state = obs[:2].detach()
                state_xs.append(state[0])
                state_ys.append(state[1])
            plt.xlim(-1, 4)
            plt.ylim(-1, 4)
            plt.plot(state_xs, state_ys, marker="o", label=success)
            plt.scatter([goal[0]], [goal[1]], color="red")
            plt.scatter(
                [pillar[0], pillar[0], pillar[0] + 1.0, pillar[0] + 1.0],
                [pillar[1], pillar[1] - 1.0, pillar[1], pillar[1] - 1.0],
                color="green",
            )
            plt.title(f"{args.method} w/ {exp_name_arch} Auto-Only Rollout")
            plt.legend()
            save_path = os.path.join(exp_dir, f"N{N}_rollout{i}.png")
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
            plt.clf()


if __name__ == "__main__":
    args = parse_args()
    main(args)
