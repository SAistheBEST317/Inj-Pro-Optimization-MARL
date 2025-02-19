
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
# from torch.utils.tensorboard import SummaryWriter
import os 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
print(os.path.abspath(','))
import sys 
sys.path.append('G:/reinforcement_learning/DRL-code-pytorch-main/5.PPO-continuous')
import gym
import argparse
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from ppo_continuous import PPO_continuous

# import envs
import environments
import torch



# from joblib import Parallel, delayed
from contextlib import closing
# from pathos.multiprocessing import ProcessingPool as Pool

from torch.multiprocessing import Pool




class main(object):
    def __init__(self, args, env_name, number, seed):
        self.a = 1
    # env = gym.make(env_name)
        self.args = args
        self.env_name = env_name
        self.number = number 
        self.seed = seed
        # df = pd.read_excel(args.posi_path) #第一列序号，第二第三列是y,x 坐标 注意！！是y, x 坐标
        # well_posi = df.values#[n,2]
        params = {'args': args}
        self.env = gym.make(env_name, **params)
        
        self.env_evaluate = gym.make(env_name, **params)  # When evaluating the policy, we need to rebuild an environment
        # Set random seed
        # env.seed(seed)
        # env.action_space.seed(seed)
        # env_evaluate.seed(seed)
        # env_evaluate.action_space.seed(seed)
        if args.random_seed is not None:
            torch.manual_seed(args.random_seed)#：为CPU中设置种子，生成随机数；
            torch.cuda.manual_seed(args.random_seed)#：为特定GPU设置种子，生成随机数；
            torch.cuda.manual_seed_all(args.random_seed)
        # np.random.seed(seed)
        # torch.manual_seed(seed)
    
        self.args.action_dim = self.env.action_space.shape[0]
        self.args.max_action = float(self.env.action_space.high[0])
        self.args.max_episode_steps = self.args.max_steps  # Maximum number of steps per episode
        print("env={}".format(self.env_name))
        print("state_dim={}".format(args.state_dim))
        print("action_dim={}".format(args.action_dim))
        print("max_action={}".format(args.max_action))
        print("max_episode_steps={}".format(args.max_episode_steps))
    
        self.evaluate_num = 0  # Record the number of evaluations
        self.evaluate_rewards = []
        self.evaluate_rewards_add = []  # Record the rewards during the evaluating
        self.evaluate_rewards_del = []  # Record the rewards during the evaluating
        self.total_steps = 0  # Record the total steps during the training
    
        self.replay_buffer = ReplayBuffer(args)
        self.agent = PPO_continuous(args)
    
        # Build a tensorboard
        # self.writer = SummaryWriter(log_dir='runs/PPO_continuous/env_{}_{}_number_{}_seed_{}'.format(env_name, args.policy_dist, number, seed))
        
        self.state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
        if args.use_reward_norm:  # Trick 3:reward normalization
            self.reward_norm = Normalization(shape=1)
        elif args.use_reward_scaling:  # Trick 4:reward scaling
            self.reward_scaling = RewardScaling(shape=1, gamma=args.gamma)
            self.reward_norm = False
        else:
            self.reward_norm = False
            self.reward_scaling = False
        
        #记录文件
        # self.record = open('record.txt','w+')
        # self.record.close()
        self.record = 'train_record.txt'
        with open(self.record,'w+') as f:
            f.write('begin to record train history\n')
            f.flush()
            
        self.eval_record_add = 'eval_record_add.txt'
        with open(self.eval_record_add,'w+') as ef:
            ef.write('begin to record eval history\n')
            ef.flush()
        self.eval_reward_add = [] #[ep_num,times] 成图的话，需要transpose一下
        
        self.eval_record_del = 'eval_record_del.txt'
        with open(self.eval_record_del,'w+') as ef:
            ef.write('begin to record eval history\n')
            ef.flush()
        self.eval_reward_del = [] #[ep_num,times] 成图的话，需要transpose一下
        self.eval_record = 'eval_record.txt'
        with open(self.eval_record,'w+') as ef:
            ef.write('begin to record eval history\n')
            ef.flush()
        self.eval_reward = [] #[ep_num,times] 成图的话，需要transpose一下
        
    def evaluate_policy(self, args, env, agent, state_norm): #使用增加井的方式来验证
        times = 1
        evaluate_reward = 0
        one_eval_reward = []

        for _ in range(times):
            #开始记录开头
            with open(self.eval_record, 'a+') as e_record:
                e_record.write(f'start eval stage episode times{_}++++++++++++++++++++\n')
                e_record.flush()
                
            s,_ = env.reset(seed = None, options={'ith': 999, 'record' : None}, \
                            test_model = {'new_path': self.args.new_path,'n_new_well': self.args.n_new_well, 'new_well_loc':  pd.read_excel(args.new_well_loc).values, 'copy_name' : '_copy_'})
            if args.use_state_norm:
                s = state_norm(s, update=False)  # During the evaluating,update=False
            done = np.array([False for i in range(self.args.n_old_well + self.args.n_new_well)])
            truncated = np.array([False for i in range(self.args.n_old_well + self.args.n_new_well)])
            episode_reward = 0
            while (not done.all()) and (not truncated.all()):
                a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
                assert a.shape[0] == (self.args.n_old_well + self.args.n_new_well)
                if args.policy_dist == "Beta":
                    action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
                else:
                    action = a
                
                s_, r, truncated, done, _ = env.step(action)
                    
                if args.use_state_norm:
                    s_ = state_norm(s_, update=False)
                episode_reward += _['profit']
                s = s_
                
                #保存结果
                with open(self.eval_record, 'a+') as e_record:
                    e_record.write(f'action:{action.flatten()}\n step_reward is {r.flatten()}\n')
                    e_record.flush()
            #记录结尾
            with open(self.eval_record, 'a+') as e_record:
                e_record.write(f'episode_reward is {episode_reward}\n')
                e_record.write('end one episode ---------------------------\n')
                e_record.flush()
                
            evaluate_reward += episode_reward
            one_eval_reward.append(episode_reward)
        self.eval_reward.append(one_eval_reward)
        return evaluate_reward / times

    def evaluate_policy_add(self, args, env, agent, state_norm): #使用增加井的方式来验证
        times = 1
        evaluate_reward = 0
        one_eval_reward = []

        for _ in range(times):
            #开始记录开头
            with open(self.eval_record_add, 'a+') as e_record:
                e_record.write(f'start eval stage episode times{_}++++++++++++++++++++\n')
                e_record.flush()
                
            s,_ = env.reset(seed = None, options={'ith': 999, 'record' : None}, \
                            test_model = {'new_path': self.args.new_path,'n_new_well': self.args.n_new_well, 'new_well_loc':  pd.read_excel(args.new_well_loc).values, 'copy_name' : '_add_'})
            if args.use_state_norm:
                s = state_norm(s, update=False)  # During the evaluating,update=False
            done = np.array([False for i in range(self.args.n_old_well + self.args.n_new_well)])
            truncated = np.array([False for i in range(self.args.n_old_well + self.args.n_new_well)])
            episode_reward = 0
            while (not done.all()) and (not truncated.all()):
                a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
                assert a.shape[0] == (self.args.n_old_well + self.args.n_new_well)
                if args.policy_dist == "Beta":
                    action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
                else:
                    action = a
                
                s_, r, truncated, done, _ = env.step(action)
                    
                if args.use_state_norm:
                    s_ = state_norm(s_, update=False)
                episode_reward += _['profit']
                s = s_
                
                #保存结果
                with open(self.eval_record_add, 'a+') as e_record:
                    e_record.write(f'action:{action.flatten()}\n step_reward is {r.flatten()}\n')
                    e_record.flush()
            #记录结尾
            with open(self.eval_record_add, 'a+') as e_record:
                e_record.write(f'episode_reward is {episode_reward}\n')
                e_record.write('end one episode ---------------------------\n')
                e_record.flush()
                
            evaluate_reward += episode_reward
            one_eval_reward.append(episode_reward)
        self.eval_reward_add.append(one_eval_reward)
        return evaluate_reward / times
    
    def evaluate_policy_del(self, args, env, agent, state_norm): #使用让一口井失效的方式来验证
        times = 1
        evaluate_reward = 0
        one_eval_reward = []

        for _ in range(times):
            #开始记录开头
            with open(self.eval_record_del, 'a+') as e_record:
                e_record.write(f'start eval stage episode times{_}++++++++++++++++++++\n')
                e_record.flush()
                
            s,_ = env.reset(seed = None, options={'ith': 999, 'record' : None}, test_model = {'new_path': self.args.path,'n_new_well': self.args.n_del_well * -1 , 'new_well_loc':  pd.read_excel(args.new_well_loc).values, 'copy_name' : '_del_'})
            if args.use_state_norm:
                s = state_norm(s, update=False)  # During the evaluating,update=False
            done = np.array([False for i in range(self.args.n_old_well - self.args.n_del_well)])
            truncated = np.array([False for i in range(self.args.n_old_well - self.args.n_del_well)])
            episode_reward = 0
            while (not done.all()) and (not truncated.all()):
                a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
                assert a.shape[0] == (self.args.n_old_well - self.args.n_del_well)
                if args.policy_dist == "Beta":
                    action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
                else:
                    action = a
                
                s_, r, truncated, done, _ = env.step(action)
                    
                if args.use_state_norm:
                    s_ = state_norm(s_, update=False)
                episode_reward += _['profit']
                s = s_
                
                #保存结果
                with open(self.eval_record_del, 'a+') as e_record:
                    e_record.write(f'action:{action.flatten()}\n step_reward is {r.flatten()}\n')
                    e_record.flush()
            #记录结尾
            with open(self.eval_record_del, 'a+') as e_record:
                e_record.write(f'episode_reward is {episode_reward}\n')
                e_record.write('end one episode ---------------------------\n')
                e_record.flush()
                
            evaluate_reward += episode_reward
            one_eval_reward.append(episode_reward)
        self.eval_reward_del.append(one_eval_reward)
        return evaluate_reward / times

    
    def pick_action(self,s):
        return self.agent.choose_action(s)
        # return 0,0.5
            
    def run_one_episode(self, n ): #函数的内置函数
    
        torch.manual_seed(n+self.iter_n)#：为CPU中设置种子，生成随机数；
        torch.cuda.manual_seed(n+self.iter_n)#：为特定GPU设置种子，生成随机数；6
        torch.cuda.manual_seed_all(n+self.iter_n)
        np.random.seed(n+self.iter_n)
        
        s,_ = self.env.reset(seed = None,options={'ith': n+10000, 'record' : None}, test_model = {'new_path': self.args.path,'n_new_well': 0 , 'new_well_loc':  pd.read_excel(args.new_well_loc).values, 'copy_name' : '_del_'})
        if self.args.use_state_norm:
            s = self.state_norm(s)
        if self.args.use_reward_scaling:
            self.reward_scaling.reset()
        done = np.array([False for i in range(self.args.n_old_well)])
        truncated = np.array([False for i in range(self.args.n_old_well)])
        episode_steps = 0
        self.env.add_step_num = 1
        s_list_single, a_list_single, a_logprob_list_single, r_list_single, s__list_single, dw_list_single, done_list_single, profit_list_single = [],[],[],[],[],[],[],[]
        while (not done.all()) and (not truncated.all()):
            episode_steps += 1
            a, a_logprob =  self.pick_action(s) # Action and the corresponding log probability [n,1]

            if self.args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * self.args.max_action  # [0,1]->[-max,max]
            else:
                action = a
            assert action.shape[0] == self.args.n_old_well,f' action dim is wrong {action.shape}, {self.args.n_old_well}'
            s_, r, truncated, done , _ = self.env.step(action) #s:[n,...]
            p = _['profit']
            
            # s_, r, done, _ = 0,0,0,0
            
            self.env.add_step_num += 1
            
            if self.args.use_state_norm:
                s_ = self.state_norm(s_)
            if self.args.use_reward_norm:
                r = self.reward_norm(r)
            elif self.args.use_reward_scaling:
                r = self.reward_scaling(r)

            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            
            # if done and episode_steps != self.args.max_episode_steps:
            #     dw = True
            # else:
            #     dw = False
            dw = truncated

                
            s_list_single.append(s)
            a_list_single.append(a)
            a_logprob_list_single.append(a_logprob)
            r_list_single.append(r)
            s__list_single.append(s_)
            dw_list_single.append(dw)
            done_list_single.append(done)
            profit_list_single.append(p)

            s = s_
            
        total_step = self.env.step_num
            
        return s_list_single, a_list_single, a_logprob_list_single, r_list_single, s__list_single, dw_list_single, done_list_single, total_step, profit_list_single
    
    def run_test(self,n):
        print(n)

            
    # def __call__(self,n):
    def __call__(self,n):
        return self.run_one_episode(n)
            
    def forward(self,n):
        self.iter_n = 0
        
        self.train_rewards = []


        while self.total_steps < self.args.max_train_steps:

        #注意，这个map是不能够调用子程序的，必须是最外层的函数才行，包括内嵌函数，类内函数等一系列都不能调用

        
            with closing(Pool(processes=n)) as pool:
                results = pool.map(self, range(n)) # 这里就会调用 call函数，
                pool.terminate()
                
                
            # self(1)

            self.iter_n += n
            with open(self.record,'a+') as record:
                for ep in results:
                    record.write('start a new episode============================================\n')
                    for num, a in enumerate(ep[1]):
                        record.write(f'action:{a.flatten()}\n step reward is {ep[3][num].flatten()}\n')
                    record.write(f'episode reward is {sum(ep[8])}\n')
                record.write('end one episode------------------------\n')
                record.flush()
                
            s_list, a_list, a_logprob_list, r_list, s__list, dw_list, done_list , step_num_list, profit_list= \
            [s for episode in results for s in  episode[0]],[a for episode in results for a in  episode[1]],\
                [a_p for episode in results for a_p in  episode[2]],\
                [r for episode in results for r in  episode[3]],\
                [s for episode in results for s in  episode[4]],\
                    [dw for episode in results for dw in  episode[5]],\
                        [d for episode in results for d in  episode[6]],\
                            [episode[7] for episode in results],\
                             [episode[8] for episode in results]
            # print('in main check shape', len(s_list),s_list[0].shape)
            # assert len(results[0][3]) >12, f'第一次ep奖励序列的是{results[0][3]},{[sum(episode[3]) for episode in results]}'
            self.train_rewards = self.train_rewards + [episode[8] for episode in results]
            
            # Take the 'action'，but store the original 'a'（especially for Beta）
            for num in range(len(s_list)):
                if (self.args.batch_size - self.replay_buffer.count) > 0:
                    self.replay_buffer.store(s_list[num], a_list[num], a_logprob_list[num], r_list[num], s__list[num], dw_list[num], done_list[num])
                else:#加满了就不加了
                    break
                
            self.total_steps += sum(step_num_list)
    
            # When the number of transitions in buffer reaches batch_size,then update network by ppo alg, and then emptify the buffer to refuse
            #当基本快要填满batchsize的时候,也就是不能再承受一次并行的经验
            if self.args.batch_size ==self.replay_buffer.count : #这里的batchsize是指的memory size，超过以后就从头开始更新
                self.agent.update(self.replay_buffer, self.total_steps)#每次填满了batchsize才开始更新一次，好费劲啊，相当于在线功能了
                self.replay_buffer.count = 0
                # eavluate after each update
                self.evaluate_num += 1
                self.evaluate_reward = self.evaluate_policy(self.args, self.env_evaluate, self.agent, self.state_norm)
                self.evaluate_rewards.append(self.evaluate_reward)
                self.evaluate_reward_add = self.evaluate_policy_add(self.args, self.env_evaluate, self.agent, self.state_norm)
                self.evaluate_rewards_add.append(self.evaluate_reward_add)
                self.evaluate_reward_del = self.evaluate_policy_del(self.args, self.env_evaluate, self.agent, self.state_norm)
                self.evaluate_rewards_del.append(self.evaluate_reward_del)
                print("evaluate_num:{} \t evaluate_reward:{}{}{} \t".format(self.evaluate_num, self.evaluate_rewards, self.evaluate_reward_add, self.evaluate_reward_del))
                # self.writer.add_scalar('step_rewards_{}'.format(self.env_name), self.evaluate_rewards[-1], global_step=self.total_steps)
                # Save the rewards
                if self.evaluate_num % self.args.save_freq == 0:
                    np.save('PPO_continuous_{}_env_{}_seed_{}.npy'.format(self.args.policy_dist, self.env_name, self.seed), np.array(self.evaluate_rewards_add),\
                            np.array(self.evaluate_rewards_del),np.array(self.evaluate_rewards))
            # self.record.close()
        return self.train_rewards,self.evaluate_rewards, self.evaluate_rewards_add, self.evaluate_rewards_del
    
    
def smooth(data, sm=1):
    smooth_data = []
    if sm > 1:
        for d in data:
            z = np.ones(len(d))
            y = np.ones(sm)*1.0
            d = np.convolve(y, d, "same")/np.convolve(y, z, "same")
            smooth_data.append(d)
    return smooth_data

def plot(data, ep_num,sm,name,save=False): #data_list（如果是多次运算可以是多层list）， episode 数量， smooth的程度 从1开始
    data = list(np.array(data).squeeze())
    fig = plt.figure()
    xdata = [i for i in range(ep_num)]
    linestyle = ['-', '--', ':', '-.']
    color = ['r', 'g', 'b', 'k']
    label = ['algo1', 'algo2', 'algo3', 'algo4']
    
    # for i in range():    
    #     print(data[i],'+++++++++++++++++++++++')
    
    plt.plot(data, color=color[0], linestyle=linestyle[0])

    plt.ylabel("production reward", fontsize=25)
    plt.xlabel("Iteration Number", fontsize=25)
    plt.title(f"optimization performance in {name}", fontsize=30)
    if save ==True:
        plt.savefig(f'{name}_reward curve.png')
    plt.show()

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    parser.add_argument("--max_train_steps", type=int, default=int(12000), help=" Maximum number of training steps")
    # parser.add_argument("--evaluate_freq", type=int, default=100, help="Evaluate the policy every n evaluate times")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency every evaluate times")
    parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
    parser.add_argument("--batch_size", type=int, default=300, help="Batch size，就是每一次更新网络使用的数据量")
    parser.add_argument("--mini_batch_size", type=int, default=100, help="Minibatch size，在每一次训练时，有很多epoch，每次epoch从中采样minibatch训练")
    parser.add_argument("--parallel_n", type=int, default=15, help="并行数N")#*******************************
    parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")
    parser.add_argument("--random_seed", type=int, default=1, help="seed")
    
    #自定义的参数
    #循环设置

    parser.add_argument("--max_steps", default=12, type = int, help = 'number of steps in one episode')#*************
    parser.add_argument("--interval", default=60, type = int, help = 'number of days in every timestep')  #*************
    parser.add_argument('--init_step', type=int, default=1, help='重启的开始优化的最初的时间步号，默认是1.也就是完成第一个时间步')
    # parser.add_argument("--well_num", default=4, type=int, help="新井数+老井数 +一口新井井位的特征数")#***********
    parser.add_argument("--n_del_well", default=2, type=int, help="新井数+老井数 +一口新井井位的特征数")#***********  
    parser.add_argument("--n_new_well", default=2, type=int, help="新井数+老井数 +一口新井井位的特征数")#***********
    parser.add_argument("--n_old_well", default=12, type=int, help="新井数+老井数 +一口新井井位的特征数")#***********
    parser.add_argument("--well_range", default=5, type=int, help="考虑收到干扰的邻井数量")#***********
    parser.add_argument("--reward_dist", default=20, type=int, help="计算reward的邻井的范围上限")#***********
    parser.add_argument("--state_dim", default=(20,), type=tuple, help="well_range * dim_num")
    # parser.add_argument("--state_c_dim", default=4, type=int, help="场数据通道数（场数量）")
    parser.add_argument("--action_dim", default=1, type=int, help="新井数+老井数 +每一口新井井位的特征数/ 井数量")#***********

    
    parser.add_argument("--well_style", default='vertical', type = str, help = '井的类型 horizon是水平井，vertical 是直井')#*************
    parser.add_argument("--path", default='eclipse_model/EGG/EGG', type = str, help = 'eclipse model path')#*************
    parser.add_argument("--posi_path", default='eclipse_model/EGG/well_position.xlsx', type = str, help = '老井坐标')#*************
    parser.add_argument("--new_well_loc", default='eclipse_model/EGG/new_well_loc.xlsx', type = str, help = '老井坐标')#*************
    parser.add_argument("--new_path", default='eclipse_model/EGG/EGG', type = str, help = 'eclipse model path')#*************
    parser.add_argument("--action_record", default='save/action_record', type = str, help = '记录action和初始状态,（不要带后缀）')#*************
    parser.add_argument("--well_list", default=['INJECT1','INJECT2','INJECT3','INJECT4','INJECT5','INJECT6','INJECT7','INJECT8','PROD1','PROD2','PROD3','PROD4'], type = list, help = 'well name list when well num = NONE')


    parser.add_argument("--grid_dim", default=[7,60,60], type = list, help = 'grid dim z,y,x')  #********************
    parser.add_argument("--grid_delta", default=[5,50,50], type = list, help = 'grid delta z,y,x')  #********************
    parser.add_argument("--horizon_len_max", default=300, type = float, help = '水平井最大长度')#*************
    parser.add_argument("--horizon_dev_max", default=0.5, type = float, help = '水平井最大井斜角')#
    parser.add_argument("--horizon_azi_max", default=2, type = float, help = '水平井最大方位角')#
    # parser.add_argument("--target", default=120000, type =float, help = 'oil price')

    parser.add_argument("--oil_price", default=1300, type = int, help = 'oil price')
    parser.add_argument("--w_price", default=-3, type = int, help = 'water price')
    parser.add_argument("--inj_w_price", default=-15, type = int, help = 'water injection price')
    parser.add_argument("--inj_g_price", default=-90, type = int, help = 'gas injection price')
    parser.add_argument("--type_cost", default = -10, type = int, help = 'cost of each high level action(negtive)')
    parser.add_argument("--perf_cost", default = -10, type = int, help = 'cost of each high level action(negtive)')
    parser.add_argument("--ctl_cost", default = -0.3, type = int, help = 'cost of each high level action(negtive)')
    parser.add_argument("--control_max", default=100, type = int, help = 'max rate of injection')
    parser.add_argument("--control_min", default=-100, type = int, help = 'max rate of injection')
    parser.add_argument("--well_type_max", default=1.999, type = float, help = 'max number of type')
    parser.add_argument("--perf_max", default=1.999, type = float, help = 'max number of type')
    parser.add_argument("--pres_max", default=600, type = int, help = 'max rate of pression') 
    parser.add_argument("--pres_min", default=0, type = int, help = 'max rate of pression') 
    parser.add_argument("--sat_max", default=1, type = int, help = 'max rate of saturation') 
    parser.add_argument("--sat_min", default=0, type = int, help = 'max rate of saturation') 

    args = parser.parse_args()

    # env_name = ['BipedalWalker-v3', 'HalfCheetah-v2', 'Hopper-v2', 'Walker2d-v2']
    # env_index = 1
    main_class = main(args, env_name="develop_qt_marl_v4.0", number=1, seed=args.random_seed)
    train_r, eval_r_add , eval_r_del = main_class.forward(args.parallel_n)
    plot(train_r,len(train_r),2,'train',True)
    plot(eval_r_add,len(eval_r_add),2,'eval_add', True)
    plot(eval_r_del,len(eval_r_del),2,'eval_del', True)
    
