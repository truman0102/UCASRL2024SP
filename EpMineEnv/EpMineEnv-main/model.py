import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from utils import ReplayBuffer, hard_update, soft_update

STATE_DIM = (128, 128, 3)
ACTION_DIM = 3
MAX_ACTION = [10, 10, 3]
HIDDEN_SIZE = [256, 64]


class CNN(nn.Module):
    def __init__(self, H, W, C, device, flatten=True):
        super(CNN, self).__init__()
        self.H = H
        self.W = W
        self.C = C
        self.features = nn.Sequential(
            nn.Conv2d(C, 8, 4),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(8, 16, 3),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(16, 32, 2),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
        )
        self.flatten = flatten
        self.device = device

    def forward(self, x):
        x = self.features(x)
        if self.flatten:
            x = x.reshape(x.size(0), -1)
        return x

    def get_feature_dim(self, H=None, W=None, C=None):
        H = H if H is not None else self.H
        W = W if W is not None else self.W
        C = C if C is not None else self.C
        x = torch.randn(1, C, H, W).to(self.device)
        x = self.features(x)
        return x.shape[1:]


class Actor(nn.Module):
    def __init__(
        self, state_dim, action_dim: int, max_action, hidden_size: list, device
    ):
        super(Actor, self).__init__()
        """
        state_dim: (H, W, C)
        action_dim: int
        max_action: list of float, whose length is action_dim
        """
        assert len(max_action) == action_dim
        self.policy = nn.Sequential()
        self.policy.add_module("fc1", nn.Linear(state_dim, hidden_size[0]))
        for i in range(1, len(hidden_size)):
            self.policy.add_module(f"sigmoid{i}", nn.Sigmoid())
            self.policy.add_module(
                f"fc{i+1}", nn.Linear(hidden_size[i - 1], hidden_size[i])
            )
        self.policy.add_module("sigmoid", nn.Sigmoid())
        self.policy.add_module("out", nn.Linear(hidden_size[-1], action_dim))
        self.policy.add_module("tanh", nn.Tanh())

        self.max_action = torch.tensor(max_action, dtype=torch.float32).to(device)

    def forward(self, x):
        x = self.policy(x)
        return torch.mul(x, self.max_action)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim: int, hidden_size: list):
        super(Critic, self).__init__()
        self.q1 = nn.Sequential()
        self.q1.add_module("fc1", nn.Linear(state_dim + action_dim, hidden_size[0]))
        for i in range(1, len(hidden_size)):
            self.q1.add_module(f"sigmoid{i}", nn.Sigmoid())
            self.q1.add_module(
                f"fc{i+1}", nn.Linear(hidden_size[i - 1], hidden_size[i])
            )
        self.q1.add_module("sigmoid", nn.Sigmoid())
        self.q1.add_module("out", nn.Linear(hidden_size[-1], 1))
        self.q2 = deepcopy(self.q1)

    def forward(self, x, u):
        x = torch.cat((x, u), 1)
        x1 = self.q1(x)
        x2 = self.q2(x)
        return x1, x2

    def Q1(self, x, u):
        x = torch.cat((x, u), 1)
        x1 = self.q1(x)
        return x1


class TD3(object):
    def __init__(
        self, H, W, C, action_dim, max_action, actor_hidden, critic_hidden, **kwargs
    ):
        self.device = kwargs.get(
            "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.cnn = CNN(H, W, C, device=self.device).to(self.device)
        self.hidden_state_dim = self.cnn.get_feature_dim().numel()
        self.actor = Actor(
            self.hidden_state_dim, action_dim, max_action, actor_hidden, self.device
        ).to(self.device)
        self.actor_target = deepcopy(self.actor).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=kwargs.get("actor_lr", 3e-4)
        )
        self.critic = Critic(self.hidden_state_dim, action_dim, critic_hidden).to(
            self.device
        )
        self.critic_target = deepcopy(self.critic).to(self.device)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=kwargs.get("critic_lr", 3e-4)
        )
        self.max_action = torch.tensor(max_action, dtype=torch.float32).to(self.device)
        self.discount = kwargs.get("discount", 0.99)
        self.tau = kwargs.get("tau", 0.005)
        self.policy_noise = kwargs.get("policy_noise", 0.2)
        self.noise_clip = kwargs.get("noise_clip", 0.5)
        self.policy_freq = kwargs.get("policy_freq", 2)
        self.batch_size = kwargs.get("batch_size", 256)
        self.total_it = 0

    def predict(self, state):
        """
        state: np.ndarray or torch.Tensor, whose shape is (N, C, H, W)
        """
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        return self.actor(self.cnn(state)).detach()

    def sample(self, replay_buffer):
        state, action, reward, next_state, done = replay_buffer.sample(self.batch_size)
        state = torch.from_numpy(state).float()
        action = torch.from_numpy(action).float()
        reward = torch.from_numpy(reward).float()
        next_state = torch.from_numpy(next_state).float()
        done = torch.from_numpy(done).float()
        return state, action, reward, next_state, done

    def learn(self, replay_buffer):

        state, action, reward, next_state, done = self.sample(replay_buffer)
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (self.actor_target(self.cnn(next_state)) + noise).clamp(
                -self.max_action, self.max_action
            )
            target_Q1, target_Q2 = self.critic_target(self.cnn(next_state), next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.discount * target_Q

        current_Q1, current_Q2 = self.critic(self.cnn(state), action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(
                self.cnn(state), self.actor(self.cnn(state))
            ).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            soft_update(self.actor_target, self.actor, self.tau)
            soft_update(self.critic_target, self.critic, self.tau)

            torch.cuda.empty_cache()

            self.total_it += 1
            return actor_loss.item(), critic_loss.item()

        self.total_it += 1
        return None, critic_loss.item()

    def value(self, state, action):
        return self.critic(self.cnn(state), action).detach()


if __name__ == "__main__":

    cnn = CNN(*STATE_DIM)

    obs = np.random.rand(4, 128, 128, 3)
    obs = np.transpose(obs, (0, 3, 1, 2))
    obs = torch.from_numpy(obs).float()
    model = TD3(
        H=128,
        W=128,
        C=3,
        action_dim=ACTION_DIM,
        max_action=MAX_ACTION,
        actor_hidden=HIDDEN_SIZE,
        critic_hidden=HIDDEN_SIZE,
    )
    with torch.no_grad():
        action = model.predict(obs)
        print(action.shape)
        q1, q2 = model.value(obs, action)
        print(q1.shape, q2.shape)
