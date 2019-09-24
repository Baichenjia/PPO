import tensorflow as tf
import numpy as np
from gym.spaces import Box, Discrete
import scipy.signal

tf.enable_eager_execution()

EPS = 1e-8
layers = tf.keras.layers


def combined_shape(length, shape=None):
    if shape is None:
        return (length, )
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def gaussian_likelihood(x, mu, log_std):
    # 可以验证是高斯公式. 类似于 tfd.Normal(loc=mu, scale=exp(log_std)).prob(x)
    pre_sum = -0.5 * (((x - mu) / (tf.exp(log_std) + EPS)) ** 2 + 2 * log_std + np.log(2 * np.pi))
    return tf.reduce_sum(pre_sum, axis=1)


def discount_cumsum(x, discount):
    """
        给定一个周期获得的奖励, 计算 discount reward.  x=[x0, x1, x2],
            output=[x0 + discount * x1 + discount^2 * x2,  x1 + discount * x2, x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class MlpNetwork(tf.keras.Model):
    def __init__(self, hidden_sizes, activation=tf.nn.tanh, output_activation=None):
        super(MlpNetwork, self).__init__()
        # sizes 是一个 list, 指定了各层的维度. activation 是中间层激活函数.
        model = []
        for size in hidden_sizes[:-1]:
            model.append(layers.Dense(size, activation=activation))
        model.append(layers.Dense(hidden_sizes[-1], activation=output_activation))
        self.model = tf.keras.Sequential(model)

    def call(self, x):
        return self.model(x)


class ActorCritic(tf.keras.Model):
    def __init__(self, hidden_sizes, activation=tf.nn.tanh, output_activation=None, action_space=None):
        super(ActorCritic, self).__init__()
        self.action_space = action_space
        self.act_dim = action_space.shape[-1]
        # Init network
        self.actor_network = MlpNetwork(hidden_sizes=list(hidden_sizes)+[self.act_dim],
                                        activation=activation,
                                        output_activation=output_activation)
        self.critic_network = MlpNetwork(hidden_sizes=list(hidden_sizes)+[1],
                                         activation=activation,
                                         output_activation=output_activation)
        # 在高斯策略中需要初始化一个方差变量, shape=(act_dim,)
        if isinstance(self.action_space, Box):
            self.log_std = tf.get_variable(name='log_std', initializer=-0.5 * np.ones(self.act_dim, dtype=np.float32))
        print("init network done.")

    def gaussian_policy(self, s, a=None):
        mu = self.actor_network(s)           # (b, act_dim)
        std = tf.exp(self.log_std)           # (b, act_dim)
        # 计算动作选择的概率
        pi = mu + tf.random_normal(tf.shape(mu)) * std             # 采样一个动作 a'
        logp_pi = gaussian_likelihood(pi, mu, self.log_std)        # log pi(a'|s)
        if a is None:
            return pi, None, logp_pi
        logp = gaussian_likelihood(a, mu, self.log_std)            # log pi(a|s)
        return pi, logp, logp_pi

    def categorical_policy(self, s, a=None):
        logits = self.actor_network(s)
        logp_all = tf.nn.log_softmax(logits)                        # soft-max
        pi = tf.squeeze(tf.multinomial(logits, 1), axis=1)          # 采样一个动作 a'
        logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=self.act_dim) * logp_all, axis=1)   # log pi(a'|s)
        if a is None:
            return pi, None, logp_pi
        logp = tf.reduce_sum(tf.one_hot(a, depth=self.act_dim) * logp_all, axis=1)       # log pi(a|s)
        return pi, logp, logp_pi

    def get_critic_output(self, s):
        return tf.squeeze(self.critic_network(s), axis=1)

    def choose_action_prob(self, s, a):
        # 根据动作空间, 调用不同的策略, 输出 pi,
        pi, logp, logp_pi = None, None, None
        if isinstance(self.action_space, Box):
            pi, logp, logp_pi = self.gaussian_policy(s, a)
        elif None and isinstance(self.action_space, Discrete):
            pi, logp, logp_pi = self.categorical_policy(s, a)
        else:
            print("ERROR in ActorCritic")
        # 输出 critic 的值函数
        v = tf.squeeze(self.critic_network(s), axis=1)
        return pi, logp, logp_pi, v


# if __name__ == '__main__':
#     import gym
#     env = gym.make("HalfCheetah-v2")
#     action_space = env.action_space
#     actor_critic = ActorCritic(hidden_sizes=[64, 64], activation=tf.nn.tanh,
#                                output_activation=None, action_space=action_space)
#
#     obs_dim = env.observation_space.shape[0]
#     act_dim = env.action_space.shape[0]
#
#     test_y = actor_critic.choose_action_prob(
#         s=tf.convert_to_tensor(np.random.random((1, obs_dim)), dtype=tf.float32),
#         a=tf.convert_to_tensor(np.random.random((1, act_dim)), dtype=tf.float32))
#
#     var_counts = np.sum([int(np.prod(v.shape)) for v in actor_critic.actor_network.trainable_variables + [actor_critic.log_std]])
#     print('actor parameters: ', var_counts)
#
#     var_counts = np.sum([int(np.prod(v.shape)) for v in actor_critic.critic_network.trainable_variables])
#     print('critic parameters: ', var_counts)

    # state = tf.convert_to_tensor(np.random.random((2, 17)), dtype=tf.float32)
    # action = tf.convert_to_tensor(np.random.random((2, 6)), dtype=tf.float32)
    # pi, logp, logp_pi, v = actor_critic(state, action)
    # print("------ END -------")
    # print(pi)
    # print(logp)
    # print(logp_pi)
    # print(v)
    # print(np.sum([int(np.prod(v.shape)) for v in actor_critic.actor_network.trainable_variables]))








