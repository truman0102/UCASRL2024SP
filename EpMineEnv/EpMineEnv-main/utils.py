import time
import logging
import argparse
import random
import numpy as np
from matplotlib import pyplot as plt


class Logger:
    def __init__(self, level=logging.INFO, file_name="log.log"):
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(message)s %(levelname)s",
            filename=file_name,
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Logger initialized")

    def __call__(self, message):
        self.logger.info(message)


def plot(data, title, xlabel, ylabel, save_path=None):
    plt.plot(range(1, len(data) + 1), data)
    plt.title(title, fontsize=15)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(alpha=0.4)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument("--algo", type=str, default="ppo", help="RL algorithm")
    args.add_argument("--seed", type=int, default=0)
    args.add_argument("--port", type=int, default=2000)
    args.add_argument("--policy", type=str, default="CnnPolicy")
    args.add_argument("--lr", type=float, default=3e-4)
    args.add_argument("--gamma", type=float, default=0.99)
    args.add_argument("--batch_size", type=int, default=256)
    args.add_argument("--buffer_size", type=int, default=int(1e5))

    return args.parse_args()


def get_time_now():
    return time.strftime("%m-%d-%H:%M:%S", time.localtime())
