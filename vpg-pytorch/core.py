import numpy as np 
import scipy.signal
from gym.spaces import Box, Discrete
import gym
import torch
import torch.nn as nn 
from torch.distributions.normal import Normal 
from torch.distributions.categorical import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def combined_shape(length, shape=None):
    if shape is None:
        return (length, )
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class MlpNetwork(nn.Module):
    def __init__(self, sizes, activation, output_activation=nn.Identity):
        super(MlpNetwork, self).__init__()
        layers = []
        for j in range(len(sizes) - 1):
            # add linear layer
            layers.append(nn.Linear(sizes[j], sizes[j+1]))
            # add activation
            if j == len(sizes) - 2:
                layers.append(output_activation())
            else:
                layers.append(activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def count_vars(net):
    return np.sum([np.prod(p.shape) for p in net.parameters()])


def discount_cumsum(x, discount):
    """
        magic from rllab for computing discounted cumulative sums of vectors.
        input:   vector x, [x0, x1, x2]
        output: [x0+discount*x1+discount^2 *x2, x1+discount*x2, x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class MLPCategoricalActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        # 离散动作空间
        super().__init__()
        self.logits_net = MlpNetwork([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def log_prob(self, pi, act):
        return pi.log_prob(act)

    def forward(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)


class MLPGaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        # 连续动作空间
        super().__init__()
        self.mu = MlpNetwork([obs_dim] + list(hidden_sizes) + [act_dim], activation)
        self.log_std = torch.nn.Parameter(-0.5*torch.ones(act_dim, dtype=torch.float32))

    def log_prob(self, pi, act):
        # 从高斯策略中sample出的动作 act=(None, act_dim), 求 log_prob后shape=(None, act_dim), 需要求和变为 (None,)
        return pi.log_prob(act).sum(axis=-1)

    def forward(self, obs):
        return Normal(loc=self.mu(obs), scale=torch.exp(self.log_std))


class MLPCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        # critic: 输出是一个节点
        super().__init__()
        self.v_net = MlpNetwork([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)  # (None,1) -> (None,)


class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()
        obs_dim = observation_space.shape[0]

        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)

        if isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        # 选择动作
        with torch.no_grad():
            pi = self.pi(obs)                      # policy distribution
            a = pi.sample()                        # choose an action
            logp_a = self.pi.log_prob(pi, a)       # log prob
            v = self.v(obs)                        # value function
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]      # return the action only


if __name__ == '__main__':
    net = MlpNetwork(sizes=[10, 20, 30], activation=nn.ReLU).to(device)
    x = torch.rand(1, 10).to(device)
    y = net(x)
    print(y.shape, "\n", net)
    print("total parameters:", count_vars(net))

    print("\nMLPCategoricalActor:")
    actor = MLPCategoricalActor(obs_dim=10, act_dim=4, hidden_sizes=[20, 30], activation=nn.ReLU).to(device)
    obs = torch.rand(2, 10).to(device)
    action_distribution = actor(obs)
    action_sample = action_distribution.sample()
    action_prob = actor.log_prob(pi=action_distribution, act=action_sample)
    print("action distribution", action_distribution)
    print("action sample:", action_sample)
    print("action probability:", torch.exp(action_prob))

    print("\nMLPGaussianActor:")
    actor = MLPGaussianActor(obs_dim=10, act_dim=4, hidden_sizes=[20, 30], activation=nn.ReLU).to(device)
    obs = torch.rand(2, 10).to(device)
    action_distribution = actor(obs)
    action_sample = action_distribution.sample()
    action_prob = actor.log_prob(pi=action_distribution, act=action_sample)
    print("action distribution", action_distribution)
    print("action sample:", action_sample)
    print("action probability:", torch.exp(action_prob))

    print("\nMLPCritic:")
    critic = MLPCritic(obs_dim=10, hidden_sizes=[20, 30], activation=nn.ReLU).to(device)
    obs = torch.rand(2, 10).to(device)
    v = critic(obs)
    print("critic:", v.shape)

    print("\nActor Critic")
    env = gym.make("CartPole-v0")
    ac = MLPActorCritic(observation_space=env.observation_space, 
        action_space=env.action_space, hidden_sizes=(64, 64), activation=nn.Tanh).to(device)
    obs = torch.rand(2, env.observation_space.shape[0]).to(device)
    a, v, logp_a = ac.step(obs)
    print("action:", a, ", value:", v, ", logp_a:", logp_a)
    



