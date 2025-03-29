import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf

from pysc2.lib.actions import FunctionCall, FUNCTIONS

from rl.agents.runner import BaseRunner
from rl.common.pre_processing import Preprocessor
from rl.common.pre_processing import is_spatial_action, stack_ndarray_dicts
from rl.common.util import mask_unused_argument_samples, flatten_first_dims, flatten_first_dims_dict

class FeudalRunner(BaseRunner):

    def __init__(self, agent, envs, summary_writer, args):
        """
        Args:
             agent: A2CAgent instance.
             envs: SubprocVecEnv instance.
             summary_writer: summary writer to log episode scores.
             args: {

             }
        """

        self.agent = agent
        self.envs = envs
        self.summary_writer = summary_writer

        self.train = args.train
        self.c = args.c
        self.d = args.d
        self.T = args.steps_per_batch #T is the length of the actualy trajectory
        self.n_steps = 2 * self.c + self.T
        self.discount = args.discount

        self.preproc = Preprocessor()
        self.last_obs = self.preproc.preprocess_obs(self.envs.reset())
        # 新增：初始化每個環境的前一個 Marine 數量（假設 score_cumulative[0] 是 Marine 數量）
        self.prev_marine_counts = np.zeros(self.envs.n_envs)

        print('\n### Feudal Runner #######')
        print(f'# agent = {self.agent}')
        print(f'# train = {self.train}')
        print(f'# n_steps = {self.n_steps}')
        print(f'# discount = {self.discount}')
        print('######################\n')

        self.states = agent.initial_state # Holds the managers and workers hidden states
        self.last_c_goals =         np.zeros((self.envs.n_envs,self.c,self.d))
        self.lc_manager_outputs =   np.zeros((self.envs.n_envs,self.c,self.d))

        self.episode_counter = 1
        self.max_score = 0.0
        self.cumulative_score = 0.0


    def run_batch(self, train_summary):

        last_obs = self.last_obs
        shapes   = (self.n_steps, self.envs.n_envs)
        values   = np.zeros(np.concatenate([[2], shapes]), dtype=np.float32) #first dim: manager values, second dim: worker values
        rewards  = np.zeros(shapes, dtype=np.float32)
        dones    = np.zeros(shapes, dtype=np.float32)
        all_obs, all_actions = [], []
        mb_states = self.states #first dim: manager values, second dim: worker values
        s = np.zeros((self.n_steps, self.envs.n_envs, self.d), dtype=np.float32)
        mb_last_c_goals = np.zeros((self.n_steps, self.envs.n_envs, self.c, self.d), dtype=np.float32)
        mb_last_mo = np.zeros((self.n_steps, self.envs.n_envs, self.c, self.d), dtype=np.float32)

        for n in range(self.n_steps):
            actions, values[:,n,:], states, s[n,:,:], self.last_c_goals, self.lc_manager_outputs = self.agent.step(last_obs, self.states, self.last_c_goals, self.lc_manager_outputs)
            actions = mask_unused_argument_samples(actions)

            all_obs.append(last_obs)
            all_actions.append(actions)
            mb_last_c_goals[n,:,:,:] = self.last_c_goals
            mb_last_mo[n,:,:,:] = self.lc_manager_outputs
            pysc2_actions = actions_to_pysc2(actions, size=last_obs['screen'].shape[1:3])
            obs_raw  = self.envs.step(pysc2_actions)
            last_obs = self.preproc.preprocess_obs(obs_raw)
            
            # 根據 marine 數量的變化來計算獎勵
            marine_counts = np.array([t.observation["score_cumulative"][0] for t in obs_raw])
            bonus = np.maximum(marine_counts - self.prev_marine_counts, 0)
            max_bonus = 3.0
            bonus = np.minimum(bonus, max_bonus)    # 將 bonus 限制在最大值 3 內
            bonus_scaled = bonus / max_bonus        # 縮放到 [0,1]
            bonus_bounded = bonus_scaled * 2 - 1      # 映射到 [-1,1]
            rewards[n, :] = bonus_bounded
            self.prev_marine_counts = marine_counts.copy()

            dones[n, :] = [t.last() for t in obs_raw]
            self.states = states

            for t in obs_raw:
                if t.last():
                    self.cumulative_score += self._summarize_episode(t)

        # 呼叫 compute_returns_and_advantages 並接收 debug 資訊
        returns, returns_intr, adv_m, adv_w, debug_info = compute_returns_and_advantages(
            rewards, dones, values, s, mb_last_c_goals[:,:,-1,:],
            self.discount, self.T, self.envs.n_envs, self.c
        )
        s_diff = compute_sdiff(s, self.c, self.T, self.envs.n_envs, self.d)
        # last_c_goals = compute_last_c_goals(goals, self.envs.n_envs, self.T, self.c, self.d)
        actions = stack_and_flatten_actions(all_actions[self.c:self.c+self.T])
        obs = stack_ndarray_dicts(all_obs)
        obs = { k:obs[k][self.c:self.c+self.T] for k in obs }
        obs = flatten_first_dims_dict(obs)
        returns = flatten_first_dims(returns)
        returns_intr = flatten_first_dims(returns_intr)
        adv_m = flatten_first_dims(adv_m)
        adv_w = flatten_first_dims(adv_w)
        s_diff = flatten_first_dims(s_diff)
        mb_last_c_goals = flatten_first_dims(mb_last_c_goals[self.c:self.c+self.T])
        prep_lc_mo = flatten_first_dims(mb_last_mo[self.c:self.c+self.T])
        self.last_obs = last_obs

        if self.train:
            return self.agent.train(
                obs,
                mb_states,
                actions,
                returns, returns_intr,
                adv_m, adv_w,
                s_diff,
                mb_last_c_goals,
                prep_lc_mo,
                debug_info=debug_info,  # 將 debug_info 傳遞進去
                summary=train_summary
            )
        else:
            return None


    def get_mean_score(self):
        return self.cumulative_score / self.episode_counter


    def get_max_score(self):
        return self.max_score


    def _summarize_episode(self, timestep):
        score = timestep.observation["score_cumulative"][0]
        episode = (self.agent.get_global_step() // self.n_steps) + 1 # because global_step is zero based
        if self.summary_writer is not None:
            summary = tf.Summary()
            summary.value.add(tag='sc2/episode_score', simple_value=score)
            self.summary_writer.add_summary(summary, episode)

        print("episode %d: score = %f" % (episode, score))
        self.max_score = max(self.max_score, score)
        self.episode_counter += 1
        return score
'''
# 定義分段映射函數
def scale_reward(score):
    if score < 25:
        reward = score / 25.0 - 1.0
    else:
        reward = (score - 25.0) / 25.0
    return np.clip(reward, -1, 1)
'''

def compute_sdiff(s, c, T, nenvs, d):
    s_diff = np.zeros((T,nenvs,d))
    for t in range(T):
        s_diff[t,:,:] = s[t+2*c,:,:] - s[t+c,:,:]
    s_diff[s_diff==0] = 1e-12
    return s_diff


# def compute_last_c_goals(goals, nenvs, T, c, d):
#     last_c_g = np.zeros((T,nenvs,c,d))
#     # goal (nsteps, nenvs, d)
#     for t in range(c,c+T):
#         last_c_g[t-c] = np.transpose(goals[t-c:t], (1,0,2))
#     return last_c_g


def compute_returns_and_advantages(rewards, dones, values, s, goals, discount, T, nenvs, c):
    alpha = 0.5
    manager_discount = 0.5
    # print('s', s.shape)
    # print('goals', goals.shape)

    # Intrinsic rewards
    r_i = np.zeros((T+1,nenvs))
    # 新增
    epsilon = 1e-6
    for t in range(c,c+T+1):
        for env in range(nenvs):
            sum_cos_dists = 0.0
            for i in range(1, c+1):
                
            # 計算內在獎勵
                _s = s[t, env] - s[t-i, env]
                _g = goals[t-i, env]
                # 新增:若向量小則為0
                norm_s = np.linalg.norm(_s)
                norm_g = np.linalg.norm(_g)
                if norm_s < 1e-6 or norm_g < 1e-6:
                    continue  # 或者 continuesum_cos_dists += 0.0
    
                num = np.squeeze(np.expand_dims(_s, axis=0) @ np.expand_dims(_g, axis=1))
                den = np.linalg.norm(_s) * np.linalg.norm(_g) + epsilon
                sum_cos_dists += num / den
                
            r_i[t-c, env] = np.clip(sum_cos_dists / c, -1, 1)  # 限制在[-1, 1]之間
            
                           
            '''  
                _s,_g = s[t,env]-s[t-i,env], goals[t-i,env]
                num = np.squeeze(np.expand_dims(_s,axis=0)@np.expand_dims(_g,axis=1))
                den = np.linalg.norm(_s)*np.linalg.norm(_g)
                #print("num", num)
                #print("den", den)
                #print("res", np.divide(num, den, out=np.zeros_like(num), where=den!=0))
                sum_cos_dists += np.divide(num, den, out=np.zeros_like(num), where=den!=0)
            r_i[t-c,env] = sum_cos_dists
            '''
    # print('r_i', r_i.shape)
    r_i /= c
    # Returns
    returns = np.zeros((T+1, nenvs))
    returns_intr = np.zeros((T+1, nenvs))
    returns[-1,:] = values[0,c+T]
    returns_intr[-1,:] = values[1,c+T]  # 用 manager 的 value 來初始化 returns_intr
    
    # 以列表形式保存每個 timestep 的 debug 資料（不取平均）
    debug_manager_rewards = []
    debug_manager_terms = []
    debug_worker_rewards = []
    debug_worker_terms = []    
    
    for t in reversed(range(T)):
        
        epsilon = 1e-6
        
        mgr_rewards = rewards[t+c, :]  # manager 部分的 reward
        mgr_term = manager_discount * returns[t+1, :] * (1 - dones[t+c, :]) + epsilon # manager 部分的 R

        wrk_rewards = rewards[t+c, :]    # worker 部分的 reward
        wrk_term = alpha * r_i[t, :] + discount * returns_intr[t+1, :] * (1 - dones[t+c, :]) + epsilon # worker 部分的 R(原始)
        wrk_term = np.clip(wrk_term, -10, 10) # 限制範圍
        # 使用 tanh 將 worker term 壓縮到 [-1,1]
        wrk_term_tanh = np.tanh(wrk_term)
        
        # 用經過動態標準化的 worker term 來計算返回值
        returns[t, :] = mgr_rewards + mgr_term
        returns_intr[t, :] = wrk_rewards + wrk_term_tanh

        # 保存原始值（每個 timestep 得到的 ndarray）
        debug_manager_rewards.append(mgr_rewards)
        debug_manager_terms.append(mgr_term)
        debug_worker_rewards.append(wrk_rewards)
        debug_worker_terms.append(wrk_term_tanh)

    # 去掉最後一個 timestep（因為 returns 的最後一個值不參與 advantage 計算）
    returns = returns[:-1, :]
    returns_intr = returns_intr[:-1, :]
    adv_m = returns - values[0, c:c+T, :]
    adv_w = returns_intr - values[1, c:c+T, :]

    # 將 debug 資料以字典形式返回（此為 list，每個元素為 shape=(nenvs,)）
    debug_info = {
        "manager_rewards": debug_manager_rewards,
        "manager_terms": debug_manager_terms,
        "worker_rewards": debug_worker_rewards,
        "worker_terms": debug_worker_terms
    }

    return returns, returns_intr, adv_m, adv_w, debug_info


def actions_to_pysc2(actions, size):
    """Convert agent action representation to FunctionCall representation."""
    height, width = size
    fn_id, arg_ids = actions
    actions_list = []
    for n in range(fn_id.shape[0]):
        a_0 = fn_id[n]
        a_l = []
        for arg_type in FUNCTIONS._func_list[a_0].args:
            arg_id = arg_ids[arg_type][n]
            if is_spatial_action[arg_type]:
                arg = [arg_id % width, arg_id // height]
            else:
                arg = [arg_id]
            a_l.append(arg)
        action = FunctionCall(a_0, a_l)
        actions_list.append(action)
    return actions_list


def stack_and_flatten_actions(lst, axis=0):
    fn_id_list, arg_dict_list = zip(*lst)
    fn_id = np.stack(fn_id_list, axis=axis)
    fn_id = flatten_first_dims(fn_id)
    arg_ids = stack_ndarray_dicts(arg_dict_list, axis=axis)
    arg_ids = flatten_first_dims_dict(arg_ids)
    return (fn_id, arg_ids)
