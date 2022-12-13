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
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
from core.env import HZ, FrankaEnv
from tap import Tap


# Suppress PyGame Import Text
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame  # noqa: E402


class ArgumentParser(Tap):
    # fmt: off
    data_dir: Path = Path("demos/")                     # Path to parent directory for saving demonstrations
    task: str = "sunny-side"                            # Task ID for demonstration collection

    # Task Parameters
    max_time_per_demo: int = 15                         # Max time (in seconds) to record demo -- default = 21 seconds

    # Collection Parameters
    collection_strategy: str = "kinesthetic"            # How demos are collected :: only `kinesthetic` supported!
    resume: bool = True                                 # Resume demonstration collection (on by default)
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


def demonstrate() -> None:
    args = ArgumentParser().parse_args()

    # Make directories for "raw" recorded states, and playback RGB states...
    #   > Note: the "record" + "playback" split is use for "kinesthetic" demos for obtaining visual state w/o humans!
    demo_raw_dir = args.data_dir / args.task / "record-raw"
    demo_rgb_dir = args.data_dir / args.task / "playback-rgb"

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

    # Start recording loop
    while True:
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
        for idx in range(len(joint_qs)):
            observations.append(env.step(joint_qs[idx])[0])

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

        # Loop on valid button input...
        a, x, y = buttons.input()
        while not a and not y and not x:
            a, x, y = buttons.input()

        if not x:
            np.savez(str(demo_rgb_dir / demo_file), **playback_dict)

        # Exit...
        if y:
            break

        # Bump Index
        if not x:
            demo_index += 1


if __name__ == "__main__":
    demonstrate()
