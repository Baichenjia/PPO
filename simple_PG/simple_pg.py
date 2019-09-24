import tensorflow as tf
import numpy as np
import gym
from gym.spaces import Discrete, Box
tf.enable_eager_execution()
layers = tf.keras.layers


def reward_to_go(rews):
    # 输入 [1,2,3,4,5],  返回 [15,14,12,9,5]
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs


class PolicyNetwork(tf.keras.Model):
    def __init__(self, sizes, activation=tf.nn.tanh, output_activation=None):
        super(PolicyNetwork, self).__init__()
        # sizes 是一个 list, 指定了各层的维度. activation 是中间层激活函数.
        model = []
        for size in sizes[:-1]:
            model.append(layers.Dense(size, activation=activation))
        model.append(layers.Dense(sizes[-1], activation=output_activation))
        self.model = tf.keras.Sequential(model)

    def call(self, x):
        return self.model(x)


class Agent(tf.keras.Model):
    def __init__(self, env, hidden_sizes=[32]):
        super(Agent, self).__init__()
        # env
        self.env = env
        self.obs_dim = self.env.observation_space.shape[0]    # 4
        self.n_acts = self.env.action_space.n                 # 2
        # policy network
        self.policy_network = PolicyNetwork(hidden_sizes+[self.n_acts])

    def reset(self):
        obs = self.env.reset()
        return obs

    def choose_action(self, obs):
        assert obs.shape[-1] == self.obs_dim
        obs = tf.convert_to_tensor(obs, tf.float32)
        logits = self.policy_network(obs)          # (batch_size, n_acts)
        actions = tf.random.categorical(logits=logits, num_samples=1)     # (batch_size, 1)
        return tf.squeeze(actions, axis=1)         # (batch_size, )

    def loss_fn(self, obs_ph, act_ph, weights_ph):
        # 计算 action logits
        act_logits = self.policy_network(obs_ph)                     # (batch_size, n_acts)
        # 需要选择的动作 act_ph, 以及 R(or Q, or Advantage Function)
        action_masks = tf.one_hot(act_ph, self.n_acts)
        # log[p(a|s)]
        log_probs = tf.reduce_sum(action_masks * tf.nn.log_softmax(act_logits), axis=1)
        # 最大化 log[p(a|s)]* R
        loss = -tf.reduce_mean(weights_ph * log_probs)
        return loss


def train_one_epoch(agent, optimizer, render=False, causality_bool=True, batch_size=5000,):
    # 智能体与环境交互一个周期，然后训练一次
    batch_obs = []          # for observations
    batch_acts = []         # for actions
    batch_weights = []      # for R(tau) weighting in policy gradient
    batch_rets = []         # for measuring episode returns
    batch_lens = []         # for measuring episode lengths

    # reset episode-specific variables
    obs = agent.reset()       # first obs comes from starting distribution
    done = False              # signal from environment that episode is over
    ep_rews = []              # list for rewards accrued throughout ep

    # render first episode of each epoch
    finished_rendering_this_epoch = False

    # collect experience by acting in the environment with current policy
    while True:
        # rendering
        if (not finished_rendering_this_epoch) and render:
            agent.env.render()

        # save obs
        batch_obs.append(obs.copy())

        # act in the environment
        act = agent.choose_action(obs.reshape(1, -1))[0]    # 这里批量为1, 因此取出后为一个变量
        obs, rew, done, _ = agent.env.step(act.numpy())
        # print(rew, end=",", flush=True)

        # save action, reward
        batch_acts.append(act)
        ep_rews.append(rew)

        if done:
            # if episode is over, record info about episode
            ep_ret, ep_len = sum(ep_rews), len(ep_rews)
            batch_rets.append(ep_ret)
            batch_lens.append(ep_len)

            if not causality_bool:
                # 1. 使用整个周期的真实奖励, 不使用未来时间步奖励的 discount 回报.
                batch_weights += [ep_ret] * ep_len
            else:
                # 2. 使用周期未来的折扣回报
                batch_weights += list(reward_to_go(ep_rews))

            # reset episode-specific variables
            obs, done, ep_rews = agent.env.reset(), False, []

            # won't render again this epoch
            finished_rendering_this_epoch = True

            # end experience loop if we have enough of it
            if len(batch_obs) > batch_size:
                break

    # # Debug
    # print(np.array(batch_acts).shape, batch_acts[0])
    # print(np.array(batch_obs).shape, batch_obs[0])
    # print(np.array(batch_weights).shape, batch_weights[0])
    # print(batch_rets, "\n\n", batch_lens)

    # take a single policy gradient update step
    with tf.GradientTape() as tape:
        act_ph = tf.convert_to_tensor(np.array(batch_acts), tf.int32)            # (batch_size, 1)
        obs_ph = tf.convert_to_tensor(np.array(batch_obs), tf.float32)           # (batch_size, 4)
        weights_ph = tf.convert_to_tensor(np.array(batch_weights), tf.float32)   # (batch_size, 1)
        batch_loss = agent.loss_fn(obs_ph, act_ph, weights_ph)

    gradients = tape.gradient(batch_loss, agent.trainable_variables)
    optimizer.apply_gradients(zip(gradients, agent.trainable_variables))
    return batch_loss.numpy(), batch_rets, batch_lens


def train(env_name, epochs, lr, render=False, causality_bool=True):
    env = gym.make(env_name)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    agent = Agent(env=env)
    best_reward = 0.0
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch(agent, optimizer, render, causality_bool)
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))
        if np.mean(batch_rets) >= best_reward:
            best_reward = np.mean(batch_rets)
            agent.save_weights("simple_agent.h5")
            print("new best reward:", best_reward)


def test(env_name, render=True):
    # init model
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]  # 4
    agent = Agent(env=env)
    # load weights
    test_ac = agent.choose_action(tf.convert_to_tensor(np.random.random((3, obs_dim)), dtype=tf.float32))
    print("test action:", test_ac)
    agent.load_weights("simple_agent.h5")
    print("load weights.")

    # test
    for i in range(10):
        obs = env.reset()
        rew_sum = 0.0
        while True:
            ac = agent.choose_action(obs.reshape(1, -1))[0]
            obs, rew, done, _ = agent.env.step(ac.numpy())
            if render:
                env.render()
            rew_sum += rew
            if done:
                print("Episodic rewards:", rew_sum)
                break


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--causality_bool', type=bool, default=True)
    args = parser.parse_args()
    print('\nUsing simplest formulation of policy gradient.\n')
    train(args.env_name, args.epochs, args.lr, render=False, causality_bool=args.causality_bool)
    # test(args.env_name, render=True)


