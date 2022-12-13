"""
imitate.py

Rolls out a learned behavioral cloning policy with the actual robot, using the necessary observation space/pre-processing
scheme. Supports RGB observations (joint observations pending).
"""
from typing import List, Optional, Tuple

import numpy as np
import torch
from core.env import HZ, FrankaEnv
from core.models import ResNetBC
from tap import Tap
from torchvision.transforms import Compose, Normalize, ToTensor


# TODO :: Better checkpoint selection... 10 epochs might overfit?
# fmt: off
CHECKPOINTS = {
    # >> 6/12 @ 6:00 PM :: These did not work great... something is not right with EE control!
    # "rgb+endeff-delta": "archive/06-12/runs/resnet-bc-endeff-delta-n=10+2022-06-12-17:24/last.ckpt",
    # "rgb+endeff-delta-window": "archive/06-12/runs/resnet-bc-endeff-delta-window-n=10+2022-06-12-17:33/last.ckpt",
    # "rgb+endeff": "archive/06-12/runs/resnet-bc-endeff-n=10+2022-06-12-17:28/last.ckpt",
    # "rgb+endeff-window": "archive/06-12/runs/resnet-bc-endeff-window-n=10+2022-06-12-17:34/last.ckpt",
    # "rgb+joint": "archive/06-12/runs/resnet-bc-joint-n=10+2022-06-12-17:29/last.ckpt",

    # =>> 6/12 @ 6:00 PM :: This worked decently - we collected more demos, will train again w/ normalized actions...
    # "rgb+joint-delta": "runs/resnet-bc-joint-delta-n=10+2022-06-12-17:31/last.ckpt",

    # =>> 6/12 @ 6:10 PM :: Technically, the 20th checkpoint had higher validation loss, the 8th is the best?
    # "rgb+joint-delta": "runs/resnet-bc-joint-delta-n=10+2022-06-12-17:31/resnet-bc-joint-delta-n=10+2022-06-12-17:31+epoch=04-train_loss=0.0000-val_loss=0.0001.ckpt",
    # "rgb+joint-delta": "runs/resnet-bc-joint-delta-n=10+2022-06-12-17:31/resnet-bc-joint-delta-n=10+2022-06-12-17:31+epoch=05-train_loss=0.0000-val_loss=0.0001.ckpt",

    # =>> 6/12 @ 7:30 PM :: THIS IS A DECENT POLICY -- HAS ~50% SUCCESS RATE -- DON'T SCREW WITH THESE PARAMETERS!
    #   > Notably, << delta_scale = 2.0, do_ema = True, ema_eta = 0.5 >>
    # "rgb+joint-delta": "runs/resnet-bc-joint-delta-n=10+2022-06-12-17:31/resnet-bc-joint-delta-n=10+2022-06-12-17:31+epoch=08-train_loss=0.0000-val_loss=0.0001.ckpt",

    # =>> 6/13 @ 11:00 AM :: Add Individual Policies trained on more demos...
    # "rgb+joint-delta+n=10": "runs/resnet-bc-joint-delta-n=10-x7+2022-06-13-10:46/resnet-bc-joint-delta-n=10+2022-06-13-10:46+epoch=06-train_loss=0.0000-val_loss=0.0000.ckpt",
    # "rgb+joint-delta+n=15": "runs/resnet-bc-joint-delta-n=15-x7+2022-06-13-10:43/resnet-bc-joint-delta-n=15+2022-06-13-10:43+epoch=08-train_loss=0.0000-val_loss=0.0000.ckpt",
    # "rgb+joint-delta+n=20": "runs/resnet-bc-joint-delta-n=20-x7+2022-06-13-10:41/resnet-bc-joint-delta-n=20+2022-06-13-10:41+epoch=05-train_loss=0.0000-val_loss=0.0000.ckpt",

    # =>> 6/13 @ 1:00 PM :: This was the golden ticket! We used this policy to rollout for user studies!
    "ensemble+rgb+joint-delta+n=15": [
        "runs/resnet-bc-joint-delta-n=15-x7+2022-06-13-10:43/resnet-bc-joint-delta-n=15+2022-06-13-10:43+epoch=08-train_loss=0.0000-val_loss=0.0000.ckpt",
        "runs/resnet-bc-joint-delta-n=15-x21+2022-06-13-10:57/resnet-bc-joint-delta-n=15-x21+2022-06-13-10:57+epoch=04-train_loss=0.0000-val_loss=0.0000.ckpt",
        "runs/resnet-bc-joint-delta-n=15-x49+2022-06-13-10:58/resnet-bc-joint-delta-n=15-x49+2022-06-13-10:58+epoch=03-train_loss=0.0000-val_loss=0.0000.ckpt",
        "runs/resnet-bc-joint-delta-n=15-x81+2022-06-13-11:00/resnet-bc-joint-delta-n=15-x81+2022-06-13-11:00+epoch=03-train_loss=0.0000-val_loss=0.0000.ckpt",
        "runs/resnet-bc-joint-delta-n=15-x343+2022-06-13-11:02/resnet-bc-joint-delta-n=15-x343+2022-06-13-11:02+epoch=16-train_loss=0.0000-val_loss=0.0000.ckpt",
    ],

    # =>> 6/14 @ 5:00 AM :: User Study Policies for Rollouts (k = 5 rollouts)
    "rgb+joint-delta+n=20": "runs/joy-naive+resnet-bc-joint-delta-n=20-x7/best.ckpt",
    # "rgb+joint-delta+n=20": "runs/priya-naive+resnet-bc-joint-delta-n=20-x7/best.ckpt",
    # "rgb+joint-delta+n=20": "runs/jenn-naive+resnet-bc-joint-delta-n=20-x7/best.ckpt",
    # "rgb+joint-delta+n=20": "runs/hao-naive+resnet-bc-joint-delta-n=20-x7/best.ckpt",

    # "rgb+joint-delta+n=20": "runs/joy-sharp+resnet-bc-joint-delta-n=20-x7/best.ckpt",
    # "rgb+joint-delta+n=20": "runs/priya-sharp+resnet-bc-joint-delta-n=20-x7/best.ckpt",
    # "rgb+joint-delta+n=20": "runs/jenn-sharp+resnet-bc-joint-delta-n=20-x7/best.ckpt",
    # "rgb+joint-delta+n=20": "runs/hao-sharp+resnet-bc-joint-delta-n=20-x7/best.ckpt",
}
# fmt: on


class ArgumentParser(Tap):
    # fmt: off
    task: str = "sunny-side"
    n: Optional[int] = 20                      # Number of demonstrations the policy was trained on (default: None)
    ensemble: bool = False                     # Whether or not to use the ensemble variant of the policy...

    # Model Parameters
    state: str = "rgb"
    action: str = "joint-delta"
    proprioceptive_dim: int = 14
    action_dim: int = 7

    # Robot Controller Parameters
    gain_type: str = "stiff"

    # Action Prediction Tweaks...
    delta_scale: float = 7.0                    # Simple multiplier for action (only for delta actions)
    do_ema: bool = True                         # Whether or not to smooth poses w/ an EMA...
    ema_eta: float = 0.5                        # EMA scalar --> multiply "predicted" by this, average by (1 - eta)
    # fmt: on


class EnsembleWrapper:
    def __init__(self, checkpoints: List[str], proprioceptive_dim: int, action_dim: int) -> None:
        self.proprioceptive_dim, self.action_dim = proprioceptive_dim, action_dim
        self.models = []

        # Create each model...
        for c in checkpoints:
            m = ResNetBC(self.proprioceptive_dim, self.action_dim)
            m.load_state_dict(torch.load(c, map_location=torch.device("cpu"))["state_dict"])
            self.models.append(m)

    def eval(self):
        for m in self.models:
            m.eval()

    def __call__(self, imgs: torch.Tensor, proprios: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and std of action across the ensemble."""
        actions = []
        for m in self.models:
            actions.append(m(imgs, proprios))  # Outputs [1, action_dim] Tensor...

        # Handle stacking & return
        actions = torch.concat(actions, dim=0)
        return actions.mean(dim=0), actions.std(dim=0)


def imitate() -> None:
    args = ArgumentParser().parse_args()
    if args.state == "rgb":
        # Load model from checkpoint (onto CPU)
        model_key = f"{args.state}+{args.action}+n={args.n}" if args.n is not None else f"{args.state}+{args.action}"
        if not args.ensemble:
            model = ResNetBC(args.proprioceptive_dim, args.action_dim)
            model.load_state_dict(torch.load(CHECKPOINTS[model_key], map_location=torch.device("cpu"))["state_dict"])

        else:
            # Create Ensemble Wrapper
            model_key = "ensemble+" + model_key
            model = EnsembleWrapper(CHECKPOINTS[model_key], args.proprioceptive_dim, args.action_dim)
    else:
        raise NotImplementedError(f"Model `{args.state}` not yet implemented!")

    # Setup Transform (for RGB) Policies
    transform = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # Put model in evaluation mode...
    model.eval()

    # Get Franka Env Parameters & Setup Environment
    if "joint" in args.action:
        env = FrankaEnv(home=args.task, hz=HZ, mode="stiff", controller="joint", camera=True, grasp=True)
    else:
        env = FrankaEnv(home=args.task, hz=HZ, mode="stiff", controller="cartesian", camera=True, grasp=True)

    # Drop into an infinite loop, keep running policy ad infinitum...
    while True:
        user_input, obs = "r", None
        while user_input == "r":
            obs = env.reset(do_grasp=True)
            user_input = input(
                "Ready to run Imitation Learning. Press (r) to reset, (q) to quit, and any other key to continue..."
            )
            if user_input == "q":
                print("[*] Exiting...")
                return

        # Grab initial state (ee_pose) --> use to smooth...
        if "joint" in args.action:
            averaged_pose = obs["q"]
        elif "endeff" in args.action:
            averaged_pose = obs["ee_pose"]
        else:
            raise NotImplementedError(f"Unrecognized action space `{args.action}` not yet implemented!")

        # Execute until Keyboard Interrupts
        try:
            while True:
                with torch.no_grad():
                    if args.state == "rgb":
                        q, ee_pose = obs["q"], obs["ee_pose"]
                        img, proprio = transform(obs["rgb"]), np.concatenate([q, ee_pose], axis=0)

                        if args.action == "endeff-delta":
                            # TODO -- Hardcoded delta handling... will updateimg.unsqueeze(0), torch.from_numpy(proprio).unsqueeze(0)
                            delta = model().numpy()[0]
                            new_pose = (args.delta_scale * delta) + ee_pose

                        elif args.action == "endeff-delta-window":
                            # TODO -- Hardcoded delta handling... will update
                            delta = model(img.unsqueeze(0), torch.from_numpy(proprio).unsqueeze(0)).numpy()[0]
                            new_pose = delta + ee_pose

                        elif args.action == "endeff":
                            new_pose = model(img.unsqueeze(0), torch.from_numpy(proprio).unsqueeze(0)).numpy()[0]

                        elif args.action == "endeff-window":
                            new_pose = model(img.unsqueeze(0), torch.from_numpy(proprio).unsqueeze(0)).numpy()[0]

                        elif args.action == "joint-delta":
                            if not args.ensemble:
                                # TODO -- Hardcoded delta handling... will update
                                delta = model(img.unsqueeze(0), torch.from_numpy(proprio).unsqueeze(0)).numpy()[0]
                                new_pose = (args.delta_scale * delta) + q

                            else:
                                # Handle ensemble logic...
                                delta_mu, delta_sigma = model(img.unsqueeze(0), torch.from_numpy(proprio).unsqueeze(0))
                                delta_mu, delta_sigma = delta_mu.numpy(), delta_sigma.numpy()

                                # TODO -- Hardcoded delta handling... will update
                                new_pose = (args.delta_scale * delta_mu) + q

                                # TODO @Kanishk -- Figure out what to do with delta_sigma...

                        elif args.action == "joint":
                            new_pose = model(img.unsqueeze(0), torch.from_numpy(proprio).unsqueeze(0)).numpy()[0]

                        else:
                            raise NotImplementedError(f"Action space `{args.action}` not yet implemented!")

                    else:
                        raise NotImplementedError("Oops!")

                # If smoothing, update pose
                if args.do_ema:
                    new_pose = (args.ema_eta * new_pose) + ((1 - args.ema_eta) * averaged_pose)
                    averaged_pose = new_pose

                # Take action in environment...
                obs, _, _, _ = env.step(new_pose)

        except KeyboardInterrupt:
            pass

        # Close Environment
        env.close()

        # Move on?
        input("Next? Press any key to continue...")


if __name__ == "__main__":
    imitate()
