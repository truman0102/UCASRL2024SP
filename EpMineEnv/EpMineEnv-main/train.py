import os
import gym
import time
import numpy as np

from envs.SingleAgent.mine_toy import EpMineEnv
from stable_baselines3.common.logger import configure
from utils import Logger, plot, get_time_now, get_args

PATH_DRL = "/home/ma-user/work/EpMineEnv-main/envs/SingleAgent/MineField_Linux-0510-random/drl.x86_64"

if __name__ == "__main__":

    args = get_args()
    algo = args.algo.strip().lower()
    log_path = os.path.join("log", algo, get_time_now())
    os.makedirs(log_path, exist_ok=True)
    plot_path = os.path.join("figure", algo, get_time_now())
    os.makedirs(plot_path, exist_ok=True)
    model_path = os.path.join("model", algo, get_time_now())
    logger = Logger(file_name=os.path.join(log_path, f"{algo}.log"))
    env = EpMineEnv(
        file_name=PATH_DRL,
        no_graph=True,
        max_episode_steps=1000,
        only_image=True,
        only_state=False,
        time_scale=200,
    )
    if algo == "ppo":
        from stable_baselines3 import PPO

        model = PPO(
            policy=args.policy,
            env=env,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            gamma=args.gamma,
            seed=args.seed,
            verbose=2,
        )
    elif algo == "a2c":
        from stable_baselines3 import A2C

        model = A2C(
            policy=args.policy,
            env=env,
            learning_rate=args.lr,
            gamma=args.gamma,
            seed=args.seed,
            verbose=2,
        )
    elif algo == "sac":
        from stable_baselines3 import SAC

        model = SAC(
            policy=args.policy,
            env=env,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            gamma=args.gamma,
            seed=args.seed,
            verbose=2,
        )
    elif algo == "td3":
        from stable_baselines3 import TD3

        model = TD3(
            policy=args.policy,
            env=env,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            gamma=args.gamma,
            seed=args.seed,
            verbose=2,
        )
    elif algo == "ddpg":
        from stable_baselines3 import DDPG
        from stable_baselines3.common.noise import NormalActionNoise

        n_actions = 3
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
        )
        model = DDPG(
            policy=args.policy,
            env=env,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            gamma=args.gamma,
            seed=args.seed,
            verbose=2,
            action_noise=action_noise,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")
    logger(f"{algo} model created")
    model.set_logger(
        configure(
            log_path,
            ["log", "stdout", "csv", "tensorboard"],
        )
    )
    start = time.time()
    model.learn(total_timesteps=5e5)
    end = time.time()
    logger(f"Training time: {(end - start)/60:.2f} minutes")
    rewards = env.reward_history
    steps = [len(r) for r in rewards]
    logger(
        f"{sum(steps)} steps and {len(steps)} episodes in total, with {env.success_total} successes and success rate {env.success_total/len(steps):.2f}"
    )
    r_max = [np.max(r) for r in rewards]
    r_avg = [np.mean(r) for r in rewards]
    r_sum = [np.sum(r) for r in rewards]
    # logger(f"steps: {steps}")
    # logger(f"r_max: {r_max}")
    # logger(f"r_avg: {r_avg}")
    # logger(f"r_sum: {r_sum}")
    plot(
        steps,
        algo,
        "Episode",
        "Steps",
        os.path.join(plot_path, f"steps_{algo}_{get_time_now()}.png"),
    )
    plot(
        r_max,
        algo,
        "Episode",
        "Max Reward",
        os.path.join(plot_path, f"r_max_{algo}_{get_time_now()}.png"),
    )
    plot(
        r_avg,
        algo,
        "Episode",
        "Average Reward",
        os.path.join(plot_path, f"r_avg_{algo}_{get_time_now()}.png"),
    )
    plot(
        r_sum,
        algo,
        "Episode",
        "Sum Reward",
        os.path.join(plot_path, f"r_sum_{algo}_{get_time_now()}.png"),
    )
    model.save(os.path.join(model_path, algo))
    env.close()
