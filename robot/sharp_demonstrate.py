"""
demonstrate.py

Collect interleaved demonstrations (in the case of kinesthetic teaching) of recording a kinesthetic demo, then
deterministically playing back the demonstration to collect visual states.

References:
    - https://github.com/facebookresearch/fairo/blob/main/polymetis/examples/2_impedance_control.py
    - https://github.com/AGI-Labs/franka_control/blob/master/record.py
    - https://github.com/AGI-Labs/franka_control/blob/master/playback.py
"""
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from core.env import HZ, FrankaEnv
from core.models import ResNetBC
from imitate import CHECKPOINTS, EnsembleWrapper
from tap import Tap
from torchvision.transforms import Compose, Normalize, ToTensor


# Suppress PyGame Import Text
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame  # noqa: E402

transform = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

class ArgumentParser(Tap):
    # fmt: off
    data_dir: Path = Path("demos/")                     # Path to parent directory for saving demonstrations
    task: str = "sunny-side"                            # Task ID for demonstration collection

    # Task Parameters
    max_time_per_demo: int = 15                         # Max time (in seconds) to record demo -- default = 21 seconds

    # Collection Parameters
    collection_strategy: str = "kinesthetic"            # How demos are collected :: only `kinesthetic` supported!
    resume: bool = True                                 # Resume demonstration collection (on by default)
    name: str = "kanishk"

    # Teaching Parameters
    num_teach: int = 5                                  # Number of demonstrations to show to the user before teaching
    num_bad_show: int =  3                             # Number of bad sub-trajectories to show to the user
    max_bad_demos: int = 15
    window_size: int = 30                               # Window size for displaying bad demonstrations

    # Compatibility Parameters
    likelihood_threshold: float = 2.5e-4                   # Threshold for accepting a demonstration
    novelty_threshold: float = 3.5e-3

    # Model Parameters
    state: str = "rgb"
    action: str = "joint-delta"
    proprioceptive_dim: int = 14
    action_dim: int = 7

    n: Optional[int] = 15                             # Number of demos the policy was trained on (default: None)
    ensemble: bool = True                             # Whether or not to use the ensemble variant of the policy...
    # fmt: on


class Buttons(object):
    def __init__(self) -> None:
        pygame.init()
        self.gamepad = pygame.joystick.Joystick(0)
        self.gamepad.init()

    def input(self) -> Tuple[bool, bool, bool]:
        # Get "A", "X", "Y" Button Presses
        pygame.event.get()
        a, x, y = self.gamepad.get_button(0), self.gamepad.get_button(2), self.gamepad.get_button(3)
        return a, x, y


def play_demo(rgb_frames):
    # Play video frame by frame using cv2
    for frame in rgb_frames:
        sized_frame = cv2.resize(frame, (600, 800)) 
        cv2.imshow("Demo", sized_frame[...,::-1])
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
        time.sleep(1/40)
    cv2.destroyAllWindows()
             

def get_demo_compatibility(demo, model):
    rgbs = demo["rgb"][:-1]
    proprios = np.concatenate([demo["q"][:-1], demo["ee_pose"][:-1]], axis=1)
    x_actions = demo["delta_q"][1:]
    rgbs = torch.stack([transform(img) for img in rgbs])
    proprios = torch.Tensor(np.stack(proprios))
    x_actions = torch.Tensor(np.stack(x_actions))
    predicted_mu, predicted_std = model(rgbs, proprios)
    likelihood = ((predicted_mu - x_actions) ** 2).mean(-1).detach().numpy()
    print(predicted_std.shape)
    novelty = predicted_std.mean(-1).detach().numpy()
    return likelihood, novelty


def get_bad_demos(likelihood, novelty, args):
    bad_idx = []
    bad_lik = []
    for d in range(len(likelihood)):
        lik = likelihood[d]
        nov = novelty[d]
        if lik > args.likelihood_threshold and nov < args.novelty_threshold:
            bad_idx.append(d)
            bad_lik.append(lik)
    bad_idx = [i for i, x in sorted(zip(bad_idx, bad_lik), key=lambda x: x[1], reverse=True)]
    return bad_idx


def show_bad_demos(bad_idx, playback_dict, args):
    for d in bad_idx[: args.num_bad_show]:
        rgb_frames = playback_dict["rgb"][d - args.window_size : d + args.window_size + 1]
        play_demo(rgb_frames)
        print(f"Bad Demo: {d}")


def demonstrate() -> None:
    args = ArgumentParser().parse_args()

    # Make directories for "raw" recorded states, and playback RGB states...
    #   > Note: the "record" + "playback" split is use for "kinesthetic" demos for obtaining visual state w/o humans!
    demo_raw_dir = args.data_dir / args.task / args.name / "record-raw"
    demo_rgb_dir = args.data_dir / args.task / args.name /"playback-rgb"

    # Fail fast if demo_dirs exist --> never overwrite!
    os.makedirs(demo_raw_dir, exist_ok=args.resume)
    os.makedirs(demo_rgb_dir, exist_ok=args.resume)

    # Setup Franka environment (on real hardware!) --> starts in "record" mode; gripper only active for shared autonomy!
    env = FrankaEnv(home=args.task, hz=HZ, mode="record", camera=True)
    demo_index = 1

    # Initialize Button Controller (for Grasping & Recording Flow)
    buttons = Buttons()

    # If `resume` -- get "latest" index
    if args.resume:
        files = os.listdir(demo_rgb_dir)
        if len(files) > 0:
            demo_index = max([int(x.split("-")[-1].split(".")[0]) for x in files]) + 1

    # show demos to learn about the task
    demo_train_dir = args.data_dir / args.task / "playback-rgb"
    train_files = os.listdir(demo_train_dir)
    if demo_index == 1:
        assert len(train_files) > args.num_teach, "Not enough demonstrations to teach!"
        random.shuffle(train_files)
        for i in range(args.num_teach):
            demo_frames = np.load(demo_train_dir / f"{train_files[i]}")["rgb"]
            play_demo(demo_frames)

    model_key = f"{args.state}+{args.action}+n={args.n}" if args.n is not None else f"{args.state}+{args.action}"
    if not args.ensemble:
        raise NotImplementedError("only ensemble implemented!")
    else:
        # Create Ensemble Wrapper
        model_key = "ensemble+" + model_key
        model = EnsembleWrapper(CHECKPOINTS[model_key], args.proprioceptive_dim, args.action_dim)
    model.eval()
    # Start recording loop
    while True:
        # TODO Add practice
        print(f"[*] Starting to Record Demonstration `{demo_index}`...")
        demo_file = f"{args.task}-{datetime.now().strftime('%m-%d')}-{demo_index}.npz"

        # Set mode appropriately
        env.set_mode("record")

        # Reset environment, and wait on user input...
        env.reset()
        print(
            "Ready to record!\n"
            f"\tYou have `{args.max_time_per_demo}` secs to complete the demo, and can use (X) to stop recording.\n"
            "\tPress (Y) to reset, and (A) to start recording!\n "
        )

        # Loop on valid button input...
        a, _, y = buttons.input()
        while not a and not y:
            a, _, y = buttons.input()

        # Reset if (Y)...
        if y:
            continue

        # Go, go, go!
        print("\t=>> Started recording... press (X) to terminate recording!")

        # Drop into Recording Loop --> for `record` mode, we really only care about joint positions & gripper widths!
        joint_qs = []
        for _ in range(int(args.max_time_per_demo * HZ) - 1):
            # Get Button Input (only if True) --> handle extended button press...
            _, x, _ = buttons.input()

            # Terminate...
            if x:
                print("\tHit (X) - stopping recording...")
                break

            # Otherwise no termination, keep on recording...
            else:
                obs, _, _, _ = env.step(None)
                joint_qs.append(obs["q"])

        # Close Environment
        env.close()

        # Save "raw" demonstration just in case...
        np.savez(str(demo_raw_dir / demo_file), hz=HZ, qs=joint_qs)

        # Enter Phase 2 -- Playback
        print("[*] Entering Playback Mode - Please reset the environment to beginning and get out of the way!")

        # Set mode
        env.set_mode("stiff")

        # Reset environment, and wait on user input....
        observations = [env.reset()]
        print("\tReady to playback! Get out of the way, and hit (A) to continue... or (X) to skip playback and retry")
        a, x, _ = buttons.input()
        while not a and not x:
            a, x, _ = buttons.input()

        if x:
            continue

        # Execute Trajectory
        likelihood, novelty = [], []
        for idx in range(len(joint_qs)):
            obs = env.step(joint_qs[idx])[0]
            observations.append(obs)
            if idx == 0:
                continue
            q, ee_pose, delta_q = obs["q"], obs["ee_pose"], obs["delta_q"]
            img, proprio = transform(obs["rgb"]), np.concatenate([q, ee_pose], axis=0)
            delta_mu, delta_sigma = model(img.unsqueeze(0), torch.from_numpy(proprio).unsqueeze(0))
            delta_mu, delta_sigma = delta_mu.detach().numpy(), delta_sigma.detach().numpy()
            lik = ((delta_q-delta_mu)**2).mean()
            nov = delta_sigma.mean()
            likelihood.append(lik)
            novelty.append(nov)
            # print(f"lik: {lik.item()} nov: {nov.item()}")
            print(f"compatibility {max(0, 1-lik/args.likelihood_threshold)}")

        # Close Environment
        env.close()

        # Write full dictionary to file
        playback_dict = {k: [] for k in list(observations[0].keys())}
        for obs in observations:
            for k in playback_dict.keys():
                playback_dict[k].append(obs[k])
        playback_dict = {k: np.array(v) for k, v in playback_dict.items()}
        playback_dict["rate"] = HZ
        playback_dict["task"] = args.task

        # Move on?
        print("Next? Press (A) to continue or (Y) to quit... or (X) to retry demo and skip save")
        a, x, y = buttons.input()
        while not a and not y and not x:
            a, x, y = buttons.input()

        if x:
            continue

        # check demo
        print("Computing compatibility")
        # likelihood, novelty = get_demo_compatibility(playback_dict, model)
        print("Getting bad demos")
        bad_demos = get_bad_demos(likelihood, novelty, args)
        print(f"found {len(bad_demos)}")
        if len(bad_demos) > args.max_bad_demos:
            print(f"[*] Demonstration rejected due to {len(bad_demos)} Bad Demonstrations!")
            print("\t=>> Here is where you went wrong...")
            show_bad_demos(bad_demos, playback_dict, args)
            random.shuffle(train_files)
            print("Try and give demos of this style")
            demo_frames = np.load(demo_train_dir / f"{train_files[0]}")["rgb"]
            play_demo(demo_frames)
            continue
        print("demo accepted!!")
        np.savez(str(demo_rgb_dir / demo_file), **playback_dict)
        demo_index += 1

        if y:
            break


if __name__ == "__main__":
    demonstrate()
