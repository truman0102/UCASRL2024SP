from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)
from mlagents_envs.base_env import ActionTuple, DecisionSteps
from scipy.spatial.transform import Rotation as R
import numpy as np
from typing import Optional
import time
import random
import cv2 as cv
import gym
import os
import socket


def IsOpen(port, ip="127.0.0.1"):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = s.connect_ex((ip, int(port)))
    if result == 0:
        print("port {} is used".format(port))
        return True
    else:
        print("port {} is not used".format(port))
        return False


TEAM_NAME = "ControlEP?team=0"
AGENT_ID = 0


def warp_action(action):
    action_dict = {"{}_{}".format(TEAM_NAME, AGENT_ID): action}
    return action_dict


class EpMineEnv(gym.Env):

    def __init__(
        self,
        file_name: str,
        port: Optional[int] = 2000,
        seed: int = 0,
        work_id: int = 0,
        time_scale: float = 200,
        max_episode_steps: int = 1000,
        only_image: bool = True,
        only_state: bool = False,
        no_graph: bool = True,
    ):
        engine_configuration_channel = EngineConfigurationChannel()
        engine_configuration_channel.set_configuration_parameters(
            width=200, height=100, time_scale=time_scale
        )
        self._engine_Environment_channel = EnvironmentParametersChannel()
        self.env = None
        self.port = port
        self.work_id = work_id
        self.eng_conf_channel = engine_configuration_channel
        self.env_file_name = file_name
        self.sd = seed
        self.no_graph = no_graph
        self.max_episode_length = max_episode_steps
        self.step_num = 0
        self.only_image = only_image
        self.only_state = only_state
        self.last_dist = 0.0
        self.last_angle = 0.0
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.current_results = None
        self.catch_state = 0
        self.reward_history = []
        self.episode_rewards = None
        self.success_total = 0

    def seed(self, sd=0):
        if self.env is not None:
            self.env.close()
        worker_id = sd
        while IsOpen(self.port + worker_id):
            worker_id += 1
        self.env = UnityEnvironment(
            file_name=self.env_file_name,
            base_port=self.port,
            seed=sd,
            worker_id=worker_id,
            side_channels=[self._engine_Environment_channel, self.eng_conf_channel],
            no_graphics=self.no_graph,
        )

    @property
    def observation_space(self):
        state_space = gym.spaces.Box(
            low=-np.Inf, high=np.Inf, shape=(7,), dtype=np.float32
        )
        if self.only_image:
            image_space = gym.spaces.Box(
                low=0, high=255, shape=(128, 128, 3), dtype=np.uint8
            )
            return image_space
        elif self.only_state:
            return state_space

        return gym.spaces.Dict({"image": image_space, "state": state_space})

    @property
    def action_space(self):
        con_spc = gym.spaces.Box(
            low=np.array([-10.0, -10.0, -3.0]),
            high=np.array([10.0, 10.0, 3.0]),
            shape=(3,),
            dtype=np.float32,
        )
        return con_spc

    def decoder_results(self, results):
        org_obs = results[TEAM_NAME].obs
        img = cv.cvtColor(
            np.array(org_obs[0][AGENT_ID] * 255, dtype=np.uint8), cv.COLOR_RGB2BGR
        )
        rotation = org_obs[1][AGENT_ID][0:4]
        position = org_obs[1][AGENT_ID][4:7]
        arm_angle = org_obs[1][AGENT_ID][7]
        catching = org_obs[1][AGENT_ID][8]
        self.is_catched = org_obs[1][AGENT_ID][9]
        mineral_pose = org_obs[1][AGENT_ID][10:13]
        state = org_obs[1][AGENT_ID]
        obs = {"image": img, "state": state}
        self.catch_state = catching
        if self.only_image:
            return img
        elif self.only_state:
            return np.array(org_obs[1][AGENT_ID][:7])
        return obs

    def get_robot_pose(self, results):
        org_obs = results[TEAM_NAME].obs
        rotation = org_obs[1][AGENT_ID][0:4]
        position = org_obs[1][AGENT_ID][4:7]
        return position, rotation

    def get_mine_pose(self, results):
        org_obs = results[TEAM_NAME].obs
        mineral_pose = org_obs[1][AGENT_ID][10:13]
        return mineral_pose

    def get_dist_to_mine(self, reuslts):
        """
        Get the distance between the robot and the mine
        """
        mine_pose = self.get_mine_pose(results=reuslts)
        robot_pose = self.get_robot_pose(results=reuslts)[0]
        dist = np.sqrt(
            (mine_pose[0] - robot_pose[0]) ** 2 + (mine_pose[2] - robot_pose[2]) ** 2
        )
        return dist

    def get_angle_to_mine(self, results):
        _, rotation = self.get_robot_pose(results=results)
        r = R.from_quat(rotation)
        # angle = r.as_euler("xyz")[1]
        angle = r.as_euler("ZYX")
        return angle

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.sd = seed
        if self.env is None:
            self.seed(self.sd)
        self.step_num = 0
        self.env.reset()
        if self.episode_rewards is not None and len(self.episode_rewards) > 0:
            self.reward_history.append(self.episode_rewards)
        self.episode_rewards = []
        obs, _, _, _ = self._step()
        self.last_dist = self.get_dist_to_mine(self.current_results)
        # self.last_angle = self.get_angle_to_mine(self.current_results)
        self.yaw, self.pitch, self.roll = self.get_angle_to_mine(self.current_results)
        return obs

    def get_reward(self, results):
        reward = results[TEAM_NAME].reward[AGENT_ID]
        return reward

    def get_dense_reward(self, results):
        final_reward = results[TEAM_NAME].reward[AGENT_ID]
        current_dist = self.get_dist_to_mine(reuslts=results)
        self.dist_reward = (self.last_dist - current_dist) * 10
        dist_reward = np.clip(self.dist_reward, -1, 1)
        current_yaw, current_pitch, current_roll = self.get_angle_to_mine(results)
        self.yaw_reward = (np.abs(self.yaw) - np.abs(current_yaw)) * 0.1
        self.pitch_reward = (np.abs(self.pitch) - np.abs(current_pitch)) * 0.1
        yaw_reward = np.clip(self.yaw_reward, -0.04, 0.04)
        pitch_reward = np.clip(self.pitch_reward, -0.04, 0.04)
        angle_reward = pitch_reward
        if current_dist < 0.5:
            angle_reward += yaw_reward
            final_reward += 1.0
        final_reward += angle_reward + dist_reward - self.step_num * 0.001
        final_reward += 10 if self.is_catched else 0  #
        self.last_dist = current_dist
        self.angle_reward = angle_reward
        self.yaw, self.pitch, self.roll = current_yaw, current_pitch, current_roll
        return final_reward

    def step(self, action):
        # action: [vy, vx, vw, arm_ang, catching]
        action = [action[0], action[1], action[2], 10.0, 1.0]
        # if self.catch_state < 0.5:
        #     action[-1] = 1.0
        # else:
        #     action[-1] = 0.0
        action = ActionTuple(np.array([action], dtype=np.float32))
        action_dict = warp_action(action=action)
        toal_reward = 0.0
        for _ in range(1):
            obs, reward, done, info = self._step(action_dict=action_dict)
            self.episode_rewards.append(reward)
            toal_reward += reward
            if done:
                self.reward_history.append(self.episode_rewards)
                self.episode_rewards = []
                break
        self.step_num += 1
        return obs, toal_reward, done, info

    def _step(self, action_dict=None) -> DecisionSteps:
        all_agents = []
        for behavior_name in self.env.behavior_specs:
            # add actions
            for agent_id in self.env.get_steps(behavior_name)[0].agent_id:
                key = behavior_name + "_{}".format(agent_id)
                all_agents.append(key)
                if action_dict != None:
                    self.env.set_action_for_agent(
                        behavior_name, agent_id, action_dict[key]
                    )
        self.env.step()

        decision_result = dict()
        terminal_result = dict()
        for behavior_name in self.env.behavior_specs:
            decision_result[behavior_name], terminal_result[behavior_name] = (
                self.env.get_steps(behavior_name)
            )
        done = False
        obs = None
        info = {}
        reward = 0.0
        if len(terminal_result[TEAM_NAME]) != 0:
            done = True
            self.success_total += 1
            # info = terminal_result[TEAM_NAME]
            obs = self.decoder_results(results=terminal_result)
            reward = self.get_dense_reward(results=terminal_result)
            self.current_results = terminal_result
            robot_position = self.get_robot_pose(results=terminal_result)[0]
        else:
            # info = decision_result[behavior_name]
            obs = self.decoder_results(results=decision_result)
            reward = self.get_dense_reward(results=decision_result)
            self.current_results = decision_result
            robot_position = self.get_robot_pose(results=decision_result)[0]
        if self.step_num > self.max_episode_length:
            done = True
        #         else:
        #             if done:
        #                 reward += 10.0
        info["robot_position"] = robot_position

        return obs, reward, done, info


def main():
    env = EpMineEnv(port=3000)
    # result = env.step()  # if not, the env won't update
    obs = env.reset()
    done = False
    step = 0
    while not done:
        print(time.time())
        action = env.action_space.sample()
        # action = [0.0, 5.0, 0.0]
        # action = [hori, vert, spin, arm_ang, catching]
        obs, reward, done, info = env.step(action)
        position = info["robot_position"]
        # print(reward)
        # print(np.array(obs["image"]).shape)
        cv.imwrite("images/{}-({}, {}).png".format(step, position[0], position[2]), obs)
        print("----------------------------------------")
        step += 1


if __name__ == "__main__":
    main()
