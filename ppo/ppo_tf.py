import numpy as np
import tensorflow as tf
import gym
import core_tf
from core_tf import ActorCritic
from spinup.utils.logx import colorize


class PPOBuffer:
    """
        经验池存储经验, VPG 智能体与环境交互, 使用 GAE 计算 Advantage 函数
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        # size 是批量大小. 这些变量用于在智能体与环境交互过程中保存记忆
        self.obs_buf = np.zeros(core_tf.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core_tf.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)  # advantage, 使用 GAE 计算
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)  # target-value, critic使用的target
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        # 计算 GAE 使用的两个参数, gamma 和 lambda
        self.gamma, self.lam = gamma, lam
        # ptr 代表当前时间步, path_start_idx 代表初始时间步
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """ 将一个时间步的交互样本存储至经验池.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
            在一个周期结束后, 计算每个时间步的advantage, 加权合成 GAE-Lambda
            如果是由于失败导致周期结束, 则 last_val=0; 否则 last_val=V(s_T).
            This allows us to bootstrap the reward-to-go calculation to account
            for time-steps beyond the arbitrary episode horizon (or epoch cutoff).
        """
        path_slice = slice(self.path_start_idx, self.ptr)  # 截取从初始位置到ptr
        rews = np.append(self.rew_buf[path_slice], last_val)  # 提取奖励序列
        vals = np.append(self.val_buf[path_slice], last_val)  # 提取值函数序列

        # 计算 GAE-Lambda advantage calculation. 用于更新 actor
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core_tf.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function. 用于更新 critic
        self.ret_buf[path_slice] = core_tf.discount_cumsum(rews, self.gamma)[:-1]  # 值函数的目标使用蒙特卡洛方法

        self.path_start_idx = self.ptr
        #
        # for i in range(self.ptr-1000, self.ptr-900):
        #     print(self.ptr, self.ret_buf[i], self.val_buf[i])

    def get(self):
        """
            Call this at the end of an epoch to get all of the data from
            the buffer, with advantages appropriately normalized (shifted to have
            mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # advantage 需要使用 MPI 中进行规约.
        # adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        # self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf, self.logp_buf]


"""
Vanilla Policy Gradient
(with GAE-Lambda for advantage estimation)
"""


def update_actor(buf_tensors, actor_critic, actor_optimizer, clip_ratio, update=True):
    obs, act, adv, _, logp_old = buf_tensors
    with tf.GradientTape() as actor_tape:
        _, logp, _, _ = actor_critic.choose_action_prob(obs, act)
        # Actor
        ratio = tf.exp(logp - logp_old)  # pi(a|s) / pi_old(a|s)
        min_adv = tf.where(adv > 0, (1 + clip_ratio) * adv, (1 - clip_ratio) * adv)
        pi_loss = -tf.reduce_mean(tf.minimum(ratio * adv, min_adv))
    # update
    if update:
        actor_gradients = actor_tape.gradient(pi_loss, actor_critic.actor_network.trainable_variables+[actor_critic.log_std])
        actor_optimizer.apply_gradients(zip(
            actor_gradients, actor_critic.actor_network.trainable_variables+[actor_critic.log_std]))
    return pi_loss, logp


def update_critic(buf_tensors, actor_critic, critic_optimizer, update=True):
    obs, _, _, ret, _ = buf_tensors
    with tf.GradientTape() as critic_tape:
        v = actor_critic.get_critic_output(obs)
        v_loss = tf.reduce_mean((ret - v) ** 2)       # critic 损失
    # update
    if update:
        critic_gradients = critic_tape.gradient(v_loss, actor_critic.critic_network.trainable_variables)
        critic_optimizer.apply_gradients(zip(critic_gradients, actor_critic.critic_network.trainable_variables))
    return v_loss, v


def ppo(env_fn, ac_kwargs=dict(),                  # ac_kwargs 存储了网络结构的参数
        seed=0, steps_per_epoch=4000, epochs=50,
        gamma=0.99, lam=0.97,                      # gamma, lambda 的设置
        clip_ratio=0.2,
        pi_lr=3e-4, vf_lr=1e-3,                    # 学习率的设置
        train_pi_iters=80, train_v_iters=80, max_ep_len=1000, target_kl=0.01):

    env = env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    tf.set_random_seed(seed)
    np.random.seed(seed)

    # Main outputs from computation graph
    actor_critic = ActorCritic(ac_kwargs["hidden_sizes"], activation=tf.nn.tanh,
                               output_activation=None, action_space=env.action_space)
    test_y = actor_critic.choose_action_prob(
        s=tf.convert_to_tensor(np.random.random((1, obs_dim)), dtype=tf.float32),
        a=tf.convert_to_tensor(np.random.random((1, act_dim)), dtype=tf.float32))

    # var counts
    var_counts = np.sum([int(np.prod(v.shape)) for v in actor_critic.trainable_variables])
    print('\nNumber of parameters: ', var_counts)

    # Experience buffer
    buf = PPOBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam)

    # Optimizers
    actor_optimizer = tf.train.AdamOptimizer(learning_rate=pi_lr)
    critic_optimizer = tf.train.AdamOptimizer(learning_rate=vf_lr)

    # Reset
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    ep_ret_old, ep_len_old, best_rew = 0, 0, -10000.
    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(steps_per_epoch):
            # a.shape=(1,6), logp_pi.shape=(1,), v_t.shape=(1,)
            a, _, logp_pi, v_t = actor_critic.choose_action_prob(
                        tf.convert_to_tensor(o.reshape(1, -1), tf.float32), a=None)
            buf.store(o, a, r, v_t, logp_pi)       # save to buffer. shape 分别为 (1,6), None, (1,) (1,)

            o, r, d, _ = env.step(a[0])            # take action
            ep_ret += r
            ep_len += 1

            terminal = d or (ep_len == max_ep_len)
            if terminal or (t == steps_per_epoch - 1):
                if not terminal:
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len)
                last_val = r if d else actor_critic.get_critic_output(tf.convert_to_tensor(o.reshape(1, -1)), tf.float32)
                buf.finish_path(last_val)         # calculate advantage function and discount return
                if terminal:
                    ep_ret_old = ep_ret
                    ep_len_old = ep_len

                o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
        print("----------------------", buf.ptr)
        # update actor
        stop_iter = False
        buf_tensors = [tf.convert_to_tensor(x) for x in buf.get()]
        obs, act, adv, ret, logp_old = buf_tensors
        # 只输出损失, 不更新, 做一个初始的记录
        pi_loss_old, _ = update_actor(buf_tensors, actor_critic, actor_optimizer, clip_ratio, update=False)
        for i in range(train_pi_iters):
            pi_loss, logp = update_actor(buf_tensors, actor_critic, actor_optimizer, clip_ratio)
            # for record
            kl = tf.reduce_mean(logp_old - logp)
            if kl > 1.5 * target_kl:
                print(colorize("Early stopping at step "+str(i)+" due to reaching max kl.",
                               color='green', bold=True, highlight=False))
                stop_iter = True
                break

        # update critic multiple times
        v_loss_old, _ = update_critic(buf_tensors, actor_critic, critic_optimizer, update=False)
        for i in range(train_v_iters):
            v_loss, _ = update_critic(buf_tensors, actor_critic, critic_optimizer)

        # Log info about actor
        pi_loss, logp = update_actor(buf_tensors, actor_critic, actor_optimizer, clip_ratio, update=False)
        kl = tf.reduce_mean(logp_old - logp)
        ratio = tf.exp(logp - logp_old)  # pi(a|s) / pi_old(a|s)
        clipped = tf.logical_or(ratio > (1 + clip_ratio), ratio < (1 - clip_ratio))
        clip_frac = tf.reduce_mean(tf.cast(clipped, tf.float32))
        delta_loss_pi = pi_loss - pi_loss_old
        ent = tf.reduce_mean(-logp)
        # log info about critic
        v_loss, v = update_critic(buf_tensors, actor_critic, critic_optimizer, update=False)
        delta_loss_v = v_loss - v_loss_old

        print("\n\n---------------------------")
        print("Epoch: \t\t", epoch)
        print("EpRet: \t\t", ep_ret_old)
        print("EpLen: \t\t", ep_len_old)
        print("VVals: \t\t", np.mean(v.numpy()))
        print("TotalEnvInteracts: \t", (epoch+1)*steps_per_epoch)
        print("LossPi: \t\t", pi_loss.numpy())
        print("LossV: \t\t", v_loss.numpy())
        print("DeltaLossPi: \t\t", delta_loss_pi.numpy())
        print("DeltaLossV: \t\t", delta_loss_v.numpy())
        print("Entropy: \t\t", ent.numpy())
        print("KL: \t\t", kl.numpy())
        print('ClipFrac:  \t\t', clip_frac.numpy())
        print("StopIter: \t\t", stop_iter)

        if ep_ret_old > best_rew:
            print("new best rewards:", ep_ret_old)
            actor_critic.save_weights("actor_critic.h5")
            best_rew = ep_ret_old


def vpg_test(env_fn, ac_kwargs, test_epochs=10):
    env = env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Main outputs from computation graph
    actor_critic = ActorCritic(ac_kwargs["hidden_sizes"], activation=tf.nn.tanh,
                               output_activation=None, action_space=env.action_space)
    test_y = actor_critic.choose_action_prob(
        s=tf.convert_to_tensor(np.random.random((1, obs_dim)), dtype=tf.float32),
        a=tf.convert_to_tensor(np.random.random((1, act_dim)), dtype=tf.float32))

    actor_critic.load_weights("actor_critic.h5")
    print("load weights.")
    for epoch in range(test_epochs):
        o, r, d, ep_ret,  = env.reset(), 0, False, 0
        for t in range(1000):
            a, _, logp_pi, v_t = actor_critic.choose_action_prob(
                        tf.convert_to_tensor(o.reshape(1, -1), tf.float32), a=None)
            o, r, d, _ = env.step(a[0])            # take action
            ep_ret += r
            if d:
                break
            env.render()
        print("Episode:", epoch, ", Reward:", ep_ret)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()

    # 训练
    # ppo(lambda: gym.make(args.env), ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
    #     gamma=args.gamma, seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs)

    # 测试
    vpg_test(lambda: gym.make(args.env), ac_kwargs=dict(hidden_sizes=[args.hid] * args.l))




