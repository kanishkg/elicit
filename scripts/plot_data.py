import argparse
import os
import pickle

import matplotlib.pyplot as plt


NS = [50, 100, 200, 300, 400, 500, 750, 1000]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", type=str, default="over_under_mix_TESTING")
    parser.add_argument("--date", type=str, default="jan4")
    parser.add_argument("--environment", type=str, default="Reach2DPillar")
    parser.add_argument("--num_rollouts", type=int, default=20)
    parser.add_argument("--num_models", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    return args


def main(args):
    data = pickle.load(open(f"./data/{args.environment}/{args.data_source}.pkl", "rb"))
    data = data[299:]
    for i, demo in enumerate(data):
        if i == args.num_rollouts:
            break
        obs = demo["obs"]
        success = demo["success"]
        obs_x = [o[0] for o in obs]
        obs_y = [o[1] for o in obs]
        goal = obs[0][2:4]
        pillar = obs[0][4:]
        plt.plot(obs_x, obs_y)
        plt.scatter([goal[0]], [goal[1]], color="red")
        plt.scatter(
            [pillar[0], pillar[0], pillar[0] + 0.5, pillar[0] + 0.5],
            [pillar[1], pillar[1] - 0.5, pillar[1], pillar[1] - 0.5],
            color="green",
        )
        plt.xlim(-1, 4)
        plt.ylim(-1, 4)
        plt.title(f"{args.data_source} Data")
        save_dir = f"./data/plots/{args.data_source}"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"data{i}.png")
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
        plt.clf()


if __name__ == "__main__":
    args = parse_args()
    main(args)
