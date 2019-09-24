import numpy as np
import tensorflow as tf
import scipy.signal
from gym.spaces import Box, Discrete

EPS = 1e-8
layers = tf.keras.layers


def combined_shape(length, shape=None):
    if shape is None:
        return (length, )
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=combined_shape(None, dim))


def placeholders(*args):
    return [placeholder(dim) for dim in args]


def placeholder_from_space(space):
    if isinstance(space, Box):
        return placeholder(space.shape)
    elif isinstance(space, Discrete):
        return tf.placeholder(dtype=tf.int32, shape=(None,))
    raise NotImplementedError


def placeholders_from_spaces(*args):
    return [placeholder_from_space(space) for space in args]


def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)


def get_vars(scope=''):
    return [x for x in tf.trainable_variables() if scope in x.name]


def count_vars(scope=''):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])


def gaussian_likelihood(x, mu, log_std):
    # 可以验证是高斯公式. 类似于 tfd.Normal(loc=mu, scale=exp(log_std)).prob(x)
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)


def discount_cumsum(x, discount):
    """
    给定一个周期获得的奖励, 计算 discount reward.
        [x0, x1, x2]
        [x0 + discount * x1 + discount^2 * x2,  x1 + discount * x2, x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


"""
Policies
"""


def mlp_categorical_policy(x, a, hidden_sizes, activation, output_activation, action_space):
    # x 是观测, a 是选择的动作. 计算 log pi(a|s).  pi 是离散动作, 形式是 soft-max
    act_dim = action_space.n
    logits = mlp(x, list(hidden_sizes)+[act_dim], activation, None)     # 通过策略网络
    logp_all = tf.nn.log_softmax(logits)                                # soft-max
    pi = tf.squeeze(tf.multinomial(logits, 1), axis=1)                  # 采样一个动作 a'
    logp = tf.reduce_sum(tf.one_hot(a, depth=act_dim) * logp_all, axis=1)       # log pi(a|s)
    logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)   # log pi(a'|s)
    return pi, logp, logp_pi


def mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation, action_space):
    # x 是观测, a 是选择的动作. 计算 log pi(a|s).  pi 的形式是高斯
    act_dim = a.shape.as_list()[-1]
    # 策略网络值输出 mu, log_std 设置为一个与输入状态 s 无关的标量, 在训练中学习.
    mu = mlp(x, list(hidden_sizes)+[act_dim], activation, output_activation)
    log_std = tf.get_variable(name='log_std', initializer=-0.5*np.ones(act_dim, dtype=np.float32))
    std = tf.exp(log_std)
    # prob
    pi = mu + tf.random_normal(tf.shape(mu)) * std      # 采样一个动作 a'
    logp = gaussian_likelihood(a, mu, log_std)          # log pi(a|s)
    logp_pi = gaussian_likelihood(pi, mu, log_std)      # log pi(a'|s)
    return pi, logp, logp_pi


"""
Actor-Critics
"""


def mlp_actor_critic(x, a, hidden_sizes=(64, 64), activation=tf.tanh,
                     output_activation=None, policy=None, action_space=None):
    # 根据环境 env 的 action_space 来判断调用哪一个策略.
    # 如果动作空间离散, 则调用 mlp_categorical_policy; 如果是连续的, 则调用 mlp_gaussian_policy.
    if policy is None and isinstance(action_space, Box):
        policy = mlp_gaussian_policy
    elif policy is None and isinstance(action_space, Discrete):
        policy = mlp_categorical_policy

    # Actor 初始化
    with tf.variable_scope('pi'):
        pi, logp, logp_pi = policy(x, a, hidden_sizes, activation, output_activation, action_space)

    # Critic 初始化, 与动作空间的类型无关, 输入状态, 输出节点为1个, 代表值函数.
    with tf.variable_scope('v'):
        v = tf.squeeze(mlp(x, list(hidden_sizes)+[1], activation, None), axis=1)
    return pi, logp, logp_pi, v


