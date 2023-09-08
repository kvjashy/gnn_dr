import argparse
import copy
import json
import os

from datetime import datetime
from pathlib import Path

import json
import pyaml
import torch
import yaml
import numpy as np

from stable_baselines3.common.utils import get_device
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy

import pybullet_data
import pybullet_envs  # register pybullet envs from bullet3

import NerveNet.gym_envs.pybullet.register_disability_envs

import gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from util import LoggingCallback
global previous_position, total_distance
previous_position = None
total_distance = 0
distances_travelled = []  # To store distances traveled for each episode

algorithms = dict(A2C=A2C, PPO=PPO)


def init_evaluate(args):

    # load the config of the trained model:
    with open(args.train_output / "train_arguments.yaml") as yaml_data:
        train_arguments = yaml.load(yaml_data,
                                    Loader=yaml.FullLoader)

    model = algorithms[train_arguments["alg"]].load(
        args.train_output / "".join(train_arguments["model_name"].split(".")[:-1]), device='cpu')
    env_name = train_arguments["task_name"]

    if args.save_again:
        if "mlp_extractor_kwargs" in model.policy_kwargs:
            if "xml_assets_path" in model.policy_kwargs["mlp_extractor_kwargs"]:
                model.policy_kwargs["mlp_extractor_kwargs"]["xml_assets_path"] = str(
                    model.policy_kwargs["mlp_extractor_kwargs"]["xml_assets_path"])
                model_folder = train_arguments["experiment_name"]
                model.save(args.train_output / "model2.zip")

    # if the base environment was trained on a another system, this path might be wrong.
    # we can't easily fix this in general...
    # but in case it is just the default path to the pybullet_data we can
    base_xml_path_parts = model.policy.mlp_extractor.xml_assets_path.parents._parts
    if "pybullet_data" in base_xml_path_parts:
        model.policy.mlp_extractor.xml_assets_path.parents._parts = Path(
            pybullet_data.getDataPath()) / "mjcf"

    env = gym.make(env_name)

    if args.render:
        env.render()  # call this before env.reset, if you want a window showing the environment
    def logging_callback(local_args, globals):
        global previous_position, total_distance

        current_position = env.unwrapped.robot.body_xyz

    # If we have a previous position, we update the distance.
        if previous_position is not None:
            distance = np.linalg.norm(np.array(current_position) - np.array(previous_position))
            total_distance += distance

        previous_position = current_position

        if local_args["done"]:
            distances_travelled.append(total_distance)  # Add the total distance for this episode to the list
            i = len(local_args["episode_rewards"])
            episode_reward = local_args.get("episode_reward", None)
            # Include total_distance in the log.
            # Reset for next episode.
            total_distance = 0
            previous_position = None

        if local_args["done"]:
            i = len(local_args["episode_rewards"])
            episode_reward = local_args.get("episode_reward", None)
            episode_length = local_args.get("episode_length", None)
            if episode_reward is not None and episode_length is not None:
                print(f"Finished {i} episode with reward {episode_reward} and length {episode_length} and distance_travelled {distances_travelled}")
            else:
                print(f"Finished {i} episode")

    episode_rewards, episode_lengths = evaluate_policy(model,
                                                       env,
                                                       n_eval_episodes=args.num_episodes,
                                                       render=args.render,
                                                       deterministic=True,
                                                       return_episode_rewards=True,
                                                       callback=logging_callback)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    max_reward = np.max(episode_rewards)  # Calculate the maximum reward


    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)

    mean_distance = np.mean(distances_travelled)
    std_distance = np.std(distances_travelled)


    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"mean_length:{mean_length:.2f} +/- {std_length:.2f}")
    print(f"mean_distance:{mean_distance:.2f} +/- {std_distance:.2f}")
    print(f"max_reward:{max_reward:.2f}")

    eval_dir = args.train_output / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    np.save(eval_dir / "episode_rewards.npy", episode_rewards)
    np.save(eval_dir / "episode_lengths.npy", episode_lengths)
    np.save(eval_dir / "distances_travelled.npy", distances_travelled)



def dir_path(path):
    if os.path.isdir(path):
        return Path(path)
    else:
        raise argparse.ArgumentTypeError(
            f"readable_dir:{path} is not a valid path")


def parse_arguments():
    p = argparse.ArgumentParser()

    p.add_argument('--config', type=argparse.FileType(mode='r'))

    p.add_argument('--train_output',
                   help="The directory where the training output & configs were logged to",
                   type=dir_path,
                   default='runs/_nerve_3')

    p.add_argument("--num_episodes",
                   help="The number of episodes to run to evaluate the model",
                   type=int,
                   default=5)

    p.add_argument('--render',
                   help='Whether to render the evaluation with pybullet client',
                   type=bool,
                   default=True)

    p.add_argument('--save_again',
                   help='Whether to save the model in a way we can load it on any system',
                   type=bool,
                   default=False)

    args = p.parse_args()

    if args.config is not None:
        data = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in data.items():
            if isinstance(value, list) and arg_dict[key] is not None:
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value

    return args


if __name__ == '__main__':
    init_evaluate(parse_arguments())
