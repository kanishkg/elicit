import argparse
import os
import pickle

import matplotlib.pyplot as plt


DATA_SOURCES = ["oracle"]
NS = [50, 100, 200, 300, 400, 500, 750, 1000]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="MLP")
    parser.add_argument("--ckpt_file", type=str, default="model_4.pt")
    parser.add_argument("--date", type=str, default="dec29")
    parser.add_argument("--environment", type=str, default="Reach2D")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--method", type=str, default="BC")
    parser.add_argument("--num_models", type=int, default=1)
    parser.add_argument("--seed", type=int, default=4)

    args = parser.parse_args()
    return args


def main(args):
    exp_name_arch = args.arch if args.num_models == 1 else "Ensemble" + args.arch
    for data_source in DATA_SOURCES:
        successes = []
        for N in NS:
            exp_name = f"{args.date}/{args.environment}/{args.method}/{exp_name_arch}/{data_source}_N{N}_seed{args.seed}"
            exp_dir = os.path.join("./out", exp_name)
            obs_x = []
            obs_y = []
            total_data = 0
            for epoch in range(1):  # args.epochs):
                data_file = os.path.join(exp_dir, "train_data.pkl")  # f'data_epoch{epoch}.pkl')

                print(data_file)
                with open(data_file, "rb") as f:
                    data = pickle.load(f)
                # for demo in data:
                #    obs = demo['obs']
                #    obs_x += [obs[i][0] for i in range(len(obs))]
                #    obs_y += [obs[i][1] for i in range(len(obs))]
                #    total_data += len(obs)
                obs = data["obs"]
                obs_x += [obs[i][0] for i in range(len(obs))]
                obs_y += [obs[i][1] for i in range(len(obs))]
                total_data += len(obs)
            print(f"N={N}, total_data={total_data}, len(data)={len(data)}")
            plt.clf()
            plt.xlim(-1, 4)
            plt.ylim(-1, 4)
            plt.scatter(obs_x, obs_y)
            plt.title(f"{args.method} w/ {exp_name_arch}, Epochs={args.epochs}, N={N} Training Data")
            save_path = f"./out/{args.date}/{args.environment}/{args.method}/{exp_name_arch}/{data_source}_N{N}_seed{args.seed}/train_data_plot.png"  # eval/train_data_N{N}_seed{args.seed}.png'
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
