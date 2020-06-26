import numpy as np
import torch
import gym
import time
import core

from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs


class VPGBuffer:
    """ 存储一个周期的样本，用于计算 GAE 函数等
    """
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam 
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size
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
        path_slice = slice(self.path_start_idx, self.ptr)     # 截取从初始位置到ptr
        rews = np.append(self.rew_buf[path_slice], last_val)  # 提取奖励序列
        vals = np.append(self.val_buf[path_slice], last_val)  # 提取值函数序列

        # 计算 GAE-Lambda advantage calculation. 用于更新 actor
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function. 用于更新 critic
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]  # 值函数的目标使用蒙特卡洛方法
        self.path_start_idx = self.ptr

    def get(self):
        """ Call this at the end of an epoch to get all of the data from
            the buffer, with advantages appropriately normalized (shifted to have
            mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf, adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}


def compute_loss_pi(data, actor):
    obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

    # 计算 actor 的损失
    pi = actor(obs)                   # 离散概率分布或连续概率分布
    logp = actor.log_prob(pi, act)    # (None,)
    loss_pi = -(logp * adv).mean()    # (None,) -> 平均值

    # Useful extra info
    approx_kl = (logp_old - logp).mean().item()
    ent = pi.entropy().mean().item()
    pi_info = dict(kl=approx_kl, ent=ent)

    return loss_pi, pi_info


def compute_loss_v(data, critic):
    # 计算 critic 的损失
    obs, ret = data['obs'], data['ret']
    return ((critic(obs) - ret)**2).mean()


def update(buf, ac, train_v_iters, pi_optimizer, vf_optimizer, logger):
    # 提取一个周期的轨迹, data 是一个dict, 键包括obs, act, ret, adv, logp
    data = buf.get()    # data['obs'].shape=(4000, obs_dim), adv.shape=(4000,)

    # 计算更新之前的计算损失函数
    pi_l_old, pi_info_old = compute_loss_pi(data=data, actor=ac.pi)
    pi_l_old = pi_l_old.item()
    v_l_old = compute_loss_v(data, critic=ac.v).item()

    # 更新策略网络的参数
    pi_optimizer.zero_grad()
    loss_pi, pi_info = compute_loss_pi(data=data, actor=ac.pi)
    loss_pi.backward()
    mpi_avg_grads(ac.pi)          # average grads across MPI processes
    pi_optimizer.step()

    # 更新值网络的参数
    for i in range(train_v_iters):
        vf_optimizer.zero_grad()
        loss_v = compute_loss_v(data=data, critic=ac.v)
        loss_v.backward()
        mpi_avg_grads(ac.v)       # average grads across MPI processes
        vf_optimizer.step()

    # Log changes from update
    kl, ent = pi_info['kl'], pi_info_old['ent']
    logger.store(LossPi=pi_l_old, LossV=v_l_old,
                 KL=kl, Entropy=ent,
                 DeltaLossPi=(loss_pi.item() - pi_l_old),
                 DeltaLossV=(loss_v.item() - v_l_old))


def vpg(env, hidden_sizes, seed=0, steps_per_epoch=4000, epochs=50, gamma=0.99, 
        pi_lr=3e-4, vf_lr=1e-3, train_v_iters=80, lam=0.97, max_ep_len=1000, logger_kwargs=dict(), save_freq=10):
    """
    Vanilla Policy Gradient    (with GAE-Lambda for advantage estimation)
    Args:
        env_fn : A function which creates a copy of the environment. The environment must satisfy the OpenAI Gym API.
        actor_critic: The constructor method for a PyTorch Module with a ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
            module. The ``step`` method should accept a batch of observations and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.
            The ``pi`` module's forward call should accept a batch of observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing 
                                           | the log probability, according to 
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical: 
                                           | make sure to flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object you provided to VPG.
        seed (int): Seed for random number generators.
        steps_per_epoch (int): Number of steps of interaction (state-action pairs) for the agent and the environment in each epoch.
        epochs (int): Number of epochs of interaction (equivalent to number of policy updates) to perform.
        gamma (float): Discount factor. (Always between 0 and 1.)
        pi_lr (float): Learning rate for policy optimizer.
        vf_lr (float): Learning rate for value function optimizer.
        train_v_iters (int): Number of gradient descent steps to take on value function per epoch.
        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1, close to 1.)
        max_ep_len (int): Maximum length of trajectory / episode / rollout.
        logger_kwargs (dict): Keyword args for EpochLogger.
        save_freq (int): How often (in terms of gap between epochs) to save the current policy and value function.
    """

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # logger
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # random seeds
    seed += 1000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 环境
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # 创建模型
    ac = core.MLPActorCritic(env.observation_space, env.action_space, hidden_sizes)

    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # Set up experience buffer. 如果有多个线程，每个线程的经验池长度为 local_steps_per_epoch
    local_steps_per_epoch = int(steps_per_epoch / num_procs()) 
    buf = VPGBuffer(obs_dim, act_dim, size=local_steps_per_epoch, gamma=gamma, lam=lam)

    # optimizer
    pi_optimizer = torch.optim.Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = torch.optim.Adam(ac.v.parameters(), lr=vf_lr)

    # setup model saving
    # logger.setup_pytorch_for_mpi()

    # interaction
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))  # (act_dim,), (), ()
            next_o, r, d, _ = env.step(a)

            ep_ret += r 
            ep_len += 1

            # save
            buf.store(o, a, r, v, logp)
            logger.store(VVals=v)

            # update obs
            o = next_o

            timeout = ep_len == max_ep_len     
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch-1
            if terminal or epoch_ended:   # timeout=True, terminal=True, epoch_ended=True/False
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0
                print("step:", t, ", done:", d, ", v:", v)
                buf.finish_path(v)
                if terminal:
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, ep_len = env.reset(), 0, 0          # 重新初始化

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)

        # Perform VPG update!
        update(buf, ac, train_v_iters, pi_optimizer, vf_optimizer, logger)

        # # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='vpg')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    
    env = gym.make(args.env)
    vpg(env, hidden_sizes=[args.hid]*args.l, gamma=args.gamma, seed=args.seed, 
        steps_per_epoch=args.steps, epochs=args.epochs, logger_kwargs=logger_kwargs)


