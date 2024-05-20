import os
import gym
import time
import argparse
import numpy as np
from utils import Logger, get_time_now
from envs.SingleAgent.mine_toy import EpMineEnv
from stable_baselines3.common.evaluation import evaluate_policy

logger = Logger(file_name="eval.log")
PATH_DRL = "/home/ma-user/work/EpMineEnv-main/envs/SingleAgent/MineField_Linux-0510-random/drl.x86_64"

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--algo", type=str, default="a2c")
    args.add_argument("--policy", type=str, default="CnnPolicy")
    args = args.parse_args()

    env = EpMineEnv(
        file_name=PATH_DRL,
        no_graph=True,
        max_episode_steps=1000,
        only_image=True,
        only_state=False,
        time_scale=200,
    )
    logger(f"Evaluation on {args.algo} algorithm at {get_time_now()}")
    if args.algo == "ppo":
        from stable_baselines3 import PPO

        model = PPO(
            policy=args.policy,
            env=env,
            verbose=2,
        )
    elif args.algo == "a2c":
        from stable_baselines3 import A2C

        model = A2C(
            policy=args.policy,
            env=env,
            verbose=2,
        )
    elif args.algo == "sac":
        from stable_baselines3 import SAC

        model = SAC(
            policy=args.policy,
            env=env,
            verbose=2,
        )
    else:
        raise ValueError("Invalid algorithm")

    MODEL_PATH = "/home/ma-user/work/EpMineEnv-main/model/a2c/05-18-09:32:07/a2c.zip"
    model.load(MODEL_PATH)
    logger(f"Model loaded from {MODEL_PATH}")
    rewards, steps = evaluate_policy(
        model, env, n_eval_episodes=10, return_episode_rewards=True
    )
    logger(steps)
