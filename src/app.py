import io
import argparse
import torch
import numpy as np
import os
import random
from base64 import b64encode
import threading
from datetime import datetime
from threading import Thread, Event



from PIL import Image
import matplotlib.pyplot as plt

from util.util import *
from util.int_utils import *
from policies import *
from web_utils.vis_utils import *
from web_utils.algorithm_wrapper import WebWrapper, DaggerWebWrapper
from web_utils.webenv_wrapper import WebCustomWrapper
from algos import *
from datasets.util import *


from flask import Flask, request, render_template, Response, session
from flask_socketio import SocketIO, emit
import wandb
import sys

global thread
thread = Thread()
thread_stop_event = Event()
sys.setrecursionlimit(10**8)
threading.stack_size(2**34)


DIR_PATH = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = os.path.join(DIR_PATH, 'templates/')

app = Flask(__name__, template_folder=TEMPLATE_PATH)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = 'qwwdils!'
socketio = SocketIO(app, async_mode='threading')


global status
status = {'frame': None, 'compatibility':.5, 'state':None, 'lock':threading.Lock()}
global input_device


def create_app(args, env, frame):
    app.config['args'] = args
    app.config['env'] = env
    app.config['frame'] = frame

def parse_args():
    parser = argparse.ArgumentParser()

    # Logging + output
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Unique experiment ID for saving/logging purposes. If not provided, date/time is used as default.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./out",
        help="Parent output directory. Files will be saved at /\{args.out_dir\}/\{args.exp_name\}.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If provided, the save directory will be overwritten even if it exists already.",
    )
    parser.add_argument("--save_iter", type=int, default=5, help="Checkpoint will be saved every args.save_iter epochs.")

    # Data loading
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/dec14_gen_oracle_reach2d_data_1k/dec14_gen_oracle_reach2d_data_1k_s4/pick-place-data-1000.pkl",
    )
    parser.add_argument("--N", type=int, default=1000, help="Size of dataset.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--low_dim", action="store_true", help="Use low-dimensional dataset.")
    # Autonomous evaluation only
    parser.add_argument(
        "--eval_only", action="store_true", help="If true, rolls out the autonomous policy of the provided trained model"
    )
    parser.add_argument(
        "--model_path", type=str, default=None, help="Path to saved model checkpoint for evaulation purposes."
    )
    parser.add_argument(
        "--N_eval_trajectories",
        type=int,
        default=100,
        help="Number of trajectories to roll out for autonomous-only evaluation.",
    )
    parser.add_argument(
        "--n_procs", type=int, default=1, help="Number of processes to use for evaluation.")

    # Environment details + rendering
    parser.add_argument("--environment", type=str, default="Reach2D", help="Environment name")
    parser.add_argument(
        "--robosuite", action="store_true", help="Whether or not the environment is a Robosuite environment"
    )

    parser.add_argument("--no_render", action="store_true", help="If true, Robosuite rendering is skipped.")
    parser.add_argument("--random_start_state", action="store_true", help="Random start state for Reach2D environment")
    parser.add_argument("--nut_type", type=str, default="round", help="Nut type for environment")
    parser.add_argument("--use_actions", action="store_true", help="If provided, actions are used in vusualizing the policy.")
    parser.add_argument("--web_interface", action="store_true", help="If provided, a web interface is started.")
    parser.add_argument("--space_mouse", action="store_true", help="If provided, space mouse is used for controlling.")

    # Method / Model details
    parser.add_argument(
        "--method", type=str, required=True, help="One of \{BC, Dagger, ThriftyDagger, HGDagger, LazyDagger\}}"
    )
    parser.add_argument("--arch", type=str, default="LinearModel", help="Model architecture to use.")
    parser.add_argument("--hidden_size", type=int, default=32, help="Hidden size of MLP if args.arch == 'MLP'")
    parser.add_argument(
        "--num_models", type=int, default=1, help="Number of models in the ensemble; if 1, a non-ensemble model is used"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--layer_norm", action="store_true", help="Whether or not to use layer normalization.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate.")
    # Dagger-specific parameters
    parser.add_argument(
        "--dagger_beta",
        type=float,
        default=0.9,
        help="DAgger parameter; policy will be (beta * expert_action) + (1-beta) * learned_policy_action",
    )
    parser.add_argument(
        "--use_indicator_beta",
        action="store_true",
        help="DAgger parameter; policy will use beta=1 for first iteration and beta=0 for following iterations.",
    )
    parser.add_argument(
        "--dagger_epochs", type=int, default=1, help="Number of expert policy interactions."
    )
    parser.add_argument(
        "--check_compatibility", action="store_true", help="Check compatibility of expert and model policies."
    )
    parser.add_argument(
        "--init_dagger_model", action="store_true", help="If provided, the dagger model is initialized from a trained BC model."
    )

    # Training parameters
    parser.add_argument("--epochs", type=int, default=5, help="Number of iterations to run overall method for")
    parser.add_argument("--clip", type=float, default=10, help="Value to clip gradients by")


    parser.add_argument("--policy", type=str, default="user", help="Specifies which expert policy to use for interactions.")
    parser.add_argument("--offline_policy", action="store_true", help="If true, uses offline policy dataset for interactions.")
    parser.add_argument("--likelihood_filter", type=float, default=1e6, help="Likelihood filter for offline policy dataset.")
    parser.add_argument("--entropy_filter", type=float, default=0.0, help="Entropy filter for offline policy dataset.")
    parser.add_argument("--normalize", action="store_true", help="Normalize observations.")

    parser.add_argument(
        "--trajectories_per_rollout",
        type=int,
        default=10,
        help=(
            "Number of trajectories to roll out per epoch, required for interactive methods and ignored for offline data"
            " methods."
        ),
    )
    parser.add_argument("--best_epoch", action="store_true", help="whether to use best epoch of the model for evaluation or the last epoch")

    # Random seed
    parser.add_argument("--seed", type=int, default=0)

    # Feedback args
    parser.add_argument("--sampling_method", type=str, default="likelihood", help="Sampling method initial demonstration.")
    parser.add_argument("--teaching_samples", type=int, default=0, help="Number of teaching samples.")
    parser.add_argument("--likelihood_threshold", type=float, default=0., help="Threshold for likelihood.")
    parser.add_argument("--entropy_threshold", type=float, default=1e6, help="Threshold for entropy.")
    parser.add_argument("--show_nearest", action="store_true", help="Whether to show the nearest demonstration.")
    parser.add_argument("--filter_method", type=str, default="sample", help="Filter method for initial demonstrations.")
    parser.add_argument("--sample_threshold", type=int, default=0, help="Threshold for sample filter.")
    parser.add_argument("--show_bad_samples", type=int, default=0, help="Number of bad samples to show for feedback.")
    parser.add_argument("--window_size", type=int, default=5, help="Window size for showing trajectory around a bad sample.")
    parser.add_argument("--online_feedback", action="store_true", help="Whether to use online feedback.")
    parser.add_argument("--img_w", type=int, default=320, help="Image width.")
    parser.add_argument("--img_h", type=int, default=256, help="Image height.")

    # thrifty dagger
    parser.add_argument("--num_q", type=int, default=2, help="Number of q funcs to use for thrifty dagger.")
    parser.add_argument("--N_initial", type=int, default=3, help="Number of initial demonstrations.")
    return parser.parse_args()

@app.route('/')
def index():
    cwd = os.getcwd()
    print(cwd)
    frame = app.config['frame']
    args = app.config['args']
    file_object = io.BytesIO()
    img = Image.fromarray((frame*255).astype('uint8'))
    img.save(file_object, 'PNG')
    base64img = "data:image/png;base64,"+b64encode(file_object.getvalue()).decode('ascii')
    # send html instructions from templates based on task
    feedback = ""
    if args.online_feedback:
        with open('src/templates/feedback.html', 'r') as f:
            feedback = f.read()
    if args.method == 'Dagger':
        if not args.online_feedback:
            with open('src/templates/instructions_dag.html', 'r') as f:
                instructions = f.read()
        else:
            with open('src/templates/instructions_sharp_dag.html', 'r') as f:
                instructions = f.read()
    if args.method == 'HGDagger':
        if not args.online_feedback:
            with open('src/templates/instructions_hgd.html', 'r') as f:
                instructions = f.read()
        else:
            with open('src/templates/instructions_sharp_hgd.html', 'r') as f:
                instructions = f.read()
    return render_template('index.html', instructions=instructions, feedback=feedback)

def send_frame():
    #infinite loop of sending frames 
    global status
    while not thread_stop_event.isSet():
        file_object = io.BytesIO()
        status["frame"].save(file_object, 'PNG')
        state = f"{status['state']}"
        base64img = "data:image/png;base64,"+b64encode(file_object.getvalue()).decode('ascii')
        cmap = plt.get_cmap('RdYlGn')
        r, g, b, a = cmap(status["compatibility"])
        compatibility_color = f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
        # get the current state of the session
        socketio.emit('response_frame', (base64img, compatibility_color, state),namespace='/', broadcast=True)
        socketio.sleep(1/30)

@socketio.on('connect')
def connect():
    global thread
    print('connected')
    if not thread.is_alive():
        print("Starting Thread")
        thread = socketio.start_background_task(send_frame)

@socketio.on('frame')
def sio_frame():
    global status
    file_object = io.BytesIO()
    # img = Image.open('frame.jpg')
    # lock status before read
    # with status["lock"]:
    status["frame"].save(file_object, 'PNG')
    state = f"{status['state']}"
    base64img = "data:image/png;base64,"+b64encode(file_object.getvalue()).decode('ascii')
    cmap = plt.get_cmap('RdYlGn')
    r, g, b, a = cmap(status["compatibility"])
    compatibility_color = f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
    # get the current state of the session
    emit('response_frame', (base64img, compatibility_color, state))

@socketio.on('keydown')
def sio_keypress(data):
    global status, input_device
    key = data
    # with input_device.lock:
    input_device.on_press(None, key, None, None, None)

@socketio.on('keyup')
def sio_keyup(data):
    global status, input_device
    key = data
    # with input_device.lock:
    input_device.on_release(None, key, None, None, None)

def init_session(args):
    global status, input_device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Set up output directories
    if args.exp_name is None:
        args.exp_name = datetime.now().strftime("%m-%d-%Y-%H:%M:%S")
    wandb_name = f"WEB-{args.environment}-{args.method}-{args.arch}-{args.num_models}-{args.hidden_size}-{args.policy}-{args.data_path.split('/')[-1].split('.')[0]}-{args.N}-{args.seed}-{args.trajectories_per_rollout}-{args.likelihood_filter}-{args.entropy_filter}"
    runid = wandb.util.generate_id()
    wandb.init(id=runid,
            name=wandb_name,
            config=args,
            entity='kanishkgandhi',
            project='sharp-dagger',
            reinit=True)

    save_dir = os.path.join(args.out_dir, args.exp_name)
    if not args.overwrite and os.path.isdir(save_dir):
        raise FileExistsError(
            f"The directory {save_dir} already exists. If you want to overwrite it, rerun with the argument --overwrite."
        )
    os.makedirs(save_dir, exist_ok=True)
    if args.environment == "PickPlace":
        env, robosuite_cfg = setup_robosuite(args, max_traj_len=PICKPLACE_MAX_TRAJ_LEN)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
    elif args.environment == "NutAssembly":
        if args.nut_type == "round":
            env, robosuite_cfg = setup_robosuite(args, max_traj_len=NUT_ASSEMBLY_MAX_TRAJ_LEN)
        elif args.nut_type == "square":
            env, robosuite_cfg = setup_robosuite(args, max_traj_len=NUT_ASSEMBLY_SQUARE_MAX_TRAJ_LEN)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
    elif args.environment == "HammerPlaceEnv":
        env, robosuite_cfg = setup_robosuite(args, max_traj_len=HAMMER_MAX_TRAJ_LEN)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
    elif args.environment == "Wipe":
        env, robosuite_cfg = setup_robosuite(args, max_traj_len=WIPE_MAX_TRAJ_LEN)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
    elif args.environment == "Door":
        env, robosuite_cfg = setup_robosuite(args, max_traj_len=DOOR_MAX_TRAJ_LEN)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
    else:
        raise NotImplementedError(f"Environment {args.environment} has not been implemented yet!")
    input_device = robosuite_cfg['input_device']
    if args.web_interface:
        obs_dim = obs_dim - 256*256*3
    # Initialize model
    model_type, model_kwargs = get_model_type_and_kwargs(args, obs_dim, act_dim)
    model = init_model(model_type, model_kwargs, device=device, num_models=args.num_models)
    model.to(device)
    # Set up method
    if args.method == "HGDagger":
        expert_policy = NutAssemblyPolicy(args.policy, env, robosuite_cfg)
        algorithm = HGDagger(model, model_kwargs, expert_policy=expert_policy, device=device, save_dir=save_dir, lr=args.lr)
        algorithm = WebWrapper(algorithm, status)
    elif args.method == "Dagger":
        expert_policy = NutAssemblyPolicy(args.policy, env, robosuite_cfg)
        algorithm = Dagger(model, model_kwargs, expert_policy=expert_policy, device=device, save_dir=save_dir, lr=args.lr)
        algorithm = DaggerWebWrapper(algorithm, status)
    elif args.method == "ThriftyDagger":
        raise NotImplementedError
        # expert_policy = get_policy(args, env, robosuite_cfg)
        # model = init_model(model_type, model_kwargs, device=device, num_models=args.num_models)
        # q_model = init_q_model(obs_dim, act_dim, args.hidden_size, args.num_q, device=device)
        # algorithm = ThriftyDagger(model, model_kwargs, expert_policy=expert_policy, q_model=q_model, device=device, save_dir=save_dir, lr=args.lr)
    else:
        raise NotImplementedError(f"Method {args.method} has not been implemented yet!")
    return env, robosuite_cfg, algorithm, runid


if __name__ == "__main__":
    args = parse_args()
    env, robosuite_cfg, algorithm, runid = init_session(args)
    curr_obs = env.reset()
    curr_obs, curr_frame = parse_obs(args, curr_obs)
    img = env.env.sim.render(
                camera_name="agentview",
                width=args.img_w,
                height=args.img_h,
                depth=False,
            )
    status["frame"] = Image.fromarray((img).astype('uint8')).transpose(method=Image.FLIP_TOP_BOTTOM)
    status["state"] = f"Practice the task and complete it {args.N_initial} times: (0/{args.N_initial}) done"

    save_dir = os.path.join(args.out_dir, args.exp_name)
    create_app(args=args, env=env, frame=curr_frame)
    # run the app in a new process
    
    # p = Process(target=lambda: socketio.run(app, host='127.0.0.1', port=6006, debug=False, use_reloader=False)).start()
    threading.Thread(target=lambda: socketio.run(app, host='127.0.0.1', port=6006, debug=False, use_reloader=False)).start()
    # skip practice if task was interrupted midway
    save_path = os.path.join(save_dir, f"data_epoch{algorithm.dagger_stage}.pkl")
    data = []
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            data = pickle.load(f)
    if data == []:
        # give user some intial experience with the task
        initial_demos = web_collect_demo(status, env, robosuite_cfg, args.N_initial, args)
    # web_visualize_demo(status, env, initial_demos[0], args)
    # get the dataset
    train, val = get_dataset(args.data_path, args.N, save_dir)
    if not args.init_dagger_model:
        status["state"] = f"The robot is training now and will soon be ready for your feedback in about 2 minutes."

    if args.method == "HGDagger":
        algorithm.run(train, val, args, env=env, robosuite_cfg=robosuite_cfg)
        status["state"] = f"<h3> The robot will now be tested without your help. The task is complete now. </h3>"
    elif args.method == "Dagger":
        algorithm.run(train, val, args, env=env, robosuite_cfg=robosuite_cfg)
        status["state"] = f"<h3> The robot will now be tested. The task is complete now. </h3>"
    else:
        raise NotImplementedError
    # eval model with the best validation loss
    # model_path = os.path.join(save_dir, "model_best.pt")
    # ckpt = torch.load(model_path, map_location=algorithm.device)
    # if args.num_models > 1:
    #     for ensemble_model, state_dict in zip(algorithm.model.models, ckpt["models"]):
    #         ensemble_model.load_state_dict(state_dict)
    # else:
    #     algorithm.model.load_state_dict(ckpt["model"])
    # algorithm.eval_auto(args, env=env, robosuite_cfg=robosuite_cfg)