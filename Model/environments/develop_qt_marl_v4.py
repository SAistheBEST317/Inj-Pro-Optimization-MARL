import os
print(os.path.abspath('.'))
import numpy as np
import gym
from gym import spaces
from gym import Env
from gym.utils import seeding
from environments.env_utils_qt_marl import write_and_run,  check_field, check_indice, npv_fun , agent_npv

from shutil import copy, copytree, rmtree
import seaborn as sns
import pandas as pd
class development_qt_marl_v4(Env):  
    #path 为不带后缀的母文件名
    def __init__(self, args = None):
        super().__init__()
        #创建一个副本，在副本上运算
        self.args = args
        f_name = args.path.split('/')[-1]
        folder_name = args.path[:-(len(f_name)+1)] #也删去中间的/符号
        new_path = folder_name + '_origin_'
        self.origin_folder = folder_name#原始文件所在文件夹
        self.creat_folder = new_path#新创建副本文件夹
        self.name = f_name
        # self.file = f_name #EGG_MODEL_ECL
        # self.folder = folder_name#G:/optimiztion_HRL/SAC_egg/egg/
        self.add_step_num = None
        self.well_style = args.well_style
        self.dim = args.grid_dim
        self.sat_max = args.sat_max
        self.pres_max = args.pres_max
        # self.control_max = liquid_max
        # self.type_max = type_max 
        # self.perf_max = perf_max #取值0，1
        self.well_list = args.well_list
        self.step_time = args.interval
        self.price = {'oil':args.oil_price,'water': args.w_price,'inj_water': args.inj_w_price,'inj_gas': args.inj_g_price}
        self._subgoals = []
        self._timed_subgoals = []
        self._tolerances = []
        # self.step_num = None
        self.max_episode_length = args.max_steps

        # if args.well_style =='vertical':
        #     self.well_posi = position
        #     print('in develop check position+++++++++++++', self.well_posi.shape)
        #     self.comp_grid = self.perf_2d_2_3d(self.well_posi, perf_start=0, perf_end = -1) #[n_old,2]
        # elif args.well_style =='horizon':
        #     add_info = np.array([[200,0,0] for i in range(position.shape[0])])
        #     self.well_posi = np.concatenate([position,add_info],axis = 1)# 从n,2 变成n，5
        #     self.comp_grid = self.perf_grid(self.well_posi, grid_dim = args.grid_dim, delt_grid = args.grid_delta)

        self.his_a_ob = None
        self.current_path = None
        # self.adj,self.adj_ind = self.cal_adj(args.well_posi_path, choose_num = args.adj_num)
        # print('in develop env check adj in init', self.adj)
        self.indice = None
        key_list = ['WLPR','WWIR','WGIR','WOPR','WWPR','WWCT','WOPT','WWPT','WWIT','FOPR','FWPR','FWIR','FOPT','FWPT','FWIT','FGIR','FGIT','WBHP','WBP9','FPR']
        #删掉重复项
        self.key_list_total = []
        for item in key_list:
            if item not in self.key_list_total:
                self.key_list_total.append(item)
#space
#1!!!!注意，这个程序里面

        self.observation_space = spaces.Box(
            low=-1, high=1, shape=( self.args.n_old_well, 3, self.args.grid_dim[0], self.args.well_range*2, self.args.well_range*2,), dtype=np.float)
            
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.args.action_dim,), dtype=np.float)
        
        # assert (self.args.state_dim[0] == self.observation_space.shape[1] and self.args.state_dim[1] == self.observation_space.shape[2]), 'args 设定的state dim 与env的ob dim不一致'
        # assert self.args.control_min<0, 'hibrid模式下 control 只表示液量因此必须小于0'
        # self.well_type = spaces.Box(#注采量范围）
        #         low = 0., 
        #         high = 1, 
        #         shape = (len(self.well_list),),
        #         dtype = np.float32)#注采类型，表示生产或者注水【0 生产，1注水，2 注气】 因为后面要整数话，所以最大值要取到接近3
        # npv_box = spaces.Box(
        #         low = 0., 
        #         high = self.npv_max, 
        #         shape = (1,),
        #         dtype = np.float32)
        
        # prof_box = spaces.Box(
        #         low = 0., 
        #         high = self.npv_max, 
        #         shape = (1,),
        #         dtype = np.float32)


 #render      
        # self.window = None

        # self.window_width = 1920
        # self.window_height = 1080
        # self.background_color = (1.0, 1.0, 1.0, 1.0)
        # self._position = (5.0, 3.0, -13.0)
        # self._lookat = (0., 0., -7.3, 0.)
#step_time()

        # # self.subgoal_radius = float(subgoal_radius)

        # self._env_geoms = ["riverbank", "water", "bridge_base", "underground"]
        # self._n_sails = 8
        

        
    def cal_adj(self,posi_path,choose_num = 3):
        well_po = pd.read_excel(posi_path,index_col = 0).values#[n,2]
        adj = np.ones((well_po.shape[0],well_po.shape[0]))#init adj
        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                adj[i,j] = np.sqrt(((well_po[i] - well_po[j])**2).sum())
        #选择N个最大值对应的索引
        neighbor = [] 
        for i in range(adj.shape[0]):
            k_maxind = adj[i].argsort()[:choose_num]#选择距离最短的n个
            neighbor.append(k_maxind)
        return adj, neighbor
        
    def inv_norm(self, x, max_value, min_value):#从【-1 - 1】 ->【0， max】
        return x * ((max_value - min_value)/2) + ((max_value + min_value)/2)
    def norm(self, x, max_value, min_value):#从【0， max】-> [-1,1]
        return (x - (max_value + min_value)/2) /((max_value - min_value)/2 + 0.001)
        

    def compute_reward(self, achieved_goal, desired_goal, info):
        #额外多出来的函数，直接等于已完成目标就可以了呀
        current = achieved_goal
        return current
    
    def get_reward(self):
        
#计算多部模拟的效益

        # print('check well indice shape in develop.py++++++++++',self.indice[0].shape, well_indice.shape, well_indice)
        profit = (self.well_indice[:self.well_num,0] * self.args.oil_price * self.args.interval \
                       + self.well_indice[:self.well_num,1] * self.args.w_price * self.args.interval \
                       + self.well_indice[:self.well_num,2] * self.args.inj_w_price * self.args.interval \
                           + self.well_indice[:self.well_num,3] * self.args.inj_g_price * self.args.interval)/1000000 #[n]
        assert profit.shape[0] == self.well_num, f'check num is {profit.shape[0]},  {self.well_num}'
        # well_type_all_oil = ((well_indice[:,1] - well_indice[:,2])>0).all()
        # well_type_all_water = ((well_indice[:,1] - well_indice[:,2])<0).all()
        
        # 采用自定义的reward来让action学习
        
        # reward = (profit.sum()>0)*1 + well_type_all_oil*-1 + well_type_all_water*-1 + max(((self.args.pres_min+100) - self.indice[self.key_list_total.index('FPR')][0]),0)*-1  \
        #     + self.new_well_near * -2 #距离进了不行
        reward = self.agg_adj_reward( self.total_well_loc,self.well_indice, self.args.reward_dist)
        # reward = profit.sum() * np.ones((self.well_num, 1))
        # print('in develop env +++++++****************+', reward, profit)
        return reward, profit.sum()#+ self.step_add_well * -10#[n,z,r,r,]        

    def get_state(self):#生成agent的记录观测量
        # print('chech obs run path',self.current_path)
    # 三维场
        run_path = self.current_path
        p_act = self.path + '.FEGRID'
        p_att = run_path + '.F'+ str(self.step_num + 1).zfill(4) #因为第0步生成F0001是查询的目标
        sat, pres, _ = check_field(p_act, p_att, dim = self.dim, squeeze= True) #第0步生成0 和1 两个，读取的应该是1，维度是(25000,)
    #井数据
        ob_keys = ['WOPR', 'WLPR', 'WWIR', 'WGIR','WBHP','WWPR','FOPR','FWIR','FPR']
        actor_ob_list = []
        include_keys = []
        for i in range(len(ob_keys)):
            actor_data = self.indice[self.key_list_total.index(ob_keys[i])] #(1,n)
            if len(actor_data.shape)==1: actor_data = actor_data.reshape(1,actor_data.shape[0]) #只有一口井的时候，维度无法保持n那个维度，所以要强制加一维保持队形
            # print('in develop checck actor shape', actor_data.shape)
            if ob_keys[i] not in include_keys:
                actor_ob_list.append(actor_data)
                include_keys.append(ob_keys[i])
        
        self.well_indice = np.stack(actor_ob_list[:6],axis = -1).squeeze(0)#(n,4)
    #场数据
        self.field_indice = np.stack(actor_ob_list[-3:],axis = -1).squeeze(0)#(1，n)
        # print(self.field_indice.shape,'+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        # print('in develop check error',padding_sat.shape,ob_range)
        if self.well_loc.shape[-1] == 0:
            self.total_well_loc = self.old_well_loc
        else:
            self.total_well_loc = np.concatenate([self.old_well_loc, self.well_loc], axis = 0)[:self.well_num]#n,2
        
        
        # well_ind = self.total_well_loc - 1#变成序号#[n,2]
        assert sat.shape[0] != 1, 'sat 有一个开头的维度'
        state = np.stack([sat.mean(0), pres.mean(0)])
        
        agent_state = self.adj_state(self.total_well_loc, self.well_indice, state, self.args.well_range).reshape(self.well_num,-1)
        assert  agent_state.shape[-1] == 20, f'{agent_state.shape},{self.args.well_range}'
        # print('++++ indevelop ', agent_state)
        return agent_state#n,2,z,r,r
    
    def get_obs(self):#生成油藏信息
    # 三维场
        run_path = self.current_path

        key_list = ['FOPR','FWPR','FWIR','FOPT','FWPT','FWIT','FGIR','FGIT']
        state_list = []
        include_keys = []
        for a in range(len(key_list)):
            data = self.indice[self.key_list_total.index(key_list[a])]#[1]
            if key_list[a] not in include_keys:
                state_list.append(data)
                include_keys.append(key_list[a])
            
        indice = np.concatenate(state_list, axis = -1) #[n,d]
        print('CHECK STATE SHAPE', indice.shape)
        
        state = indice.flatten()

        return state#[n*d]

        
    def step(self, action):# [n，1] #
        assert action.shape[-1] == 1, '环境里的action带有其他维度{action.shape}'
        #因为每一步他要先都step之前的ob所以，step的升级必须再当前步
        self.step_num += 1
    
        assert action.shape[0] == self.args.n_old_well + self.n_new_well ,f' action 维度不对，{action.shape}, {self.args.n_old_well},{self.n_new_well}'
        # assert self.args.action_dim == self.args.well_num, '设定的action dim 要等于well num‘'
        
        run_path = write_and_run(self.args, self.path, self.well_list, action.flatten(), self.comp_grid, self.step_num, self.step_time, self.args.init_step, self.record)
        if self.step_num == 0: 
            restart = False
        else:#从第1步开始，我们虽然生成第1步时还是考虑使用的原文件，但要查看的结果已经是restart文件格式了
            restart = True

        copy(run_path+'.RSM', run_path+'_c.RSM')
        self.indice = check_indice(run_path + '.RSM', time_len=1, key_words = self.key_list_total,restart = restart)
        
        self.current_path = run_path
        
        next_state = self.get_state()
        
        reward, profit = self.get_reward()
        print('单步计算的reward 和profit',reward, profit)
        #计算全局收益
        # #done的判别收益
        print('in env check reward shape', self.acc_reward.shape, reward.shape)
        self.acc_reward += reward
        #终止条件 
        #奖励达到了，或者最大步数超过了，或者不再盈利了
        # print('in develop env check npv', 'reward:',reward.shape,'judge reward is ', delt_a_ob.shape)
        truncated = True if (self.step_num - self.args.init_step)>= self.args.max_steps else False
        done = True if  truncated else False#g_reward < 0 #+ np.array([self.step_num >= self.max_episode_length for i in range(action.shape[0])]) #or partial_obs['profit'] <= 0

        
        agent_done = np.array([done for i in range(self.well_num)]).reshape(-1,1)
        agent_truncated = np.array([truncated for i in range(self.well_num)]).reshape(-1,1)#[n,1]
        assert agent_done.shape[-1] == 1, ' the last dim of done is 1'
        # print('in develop check done shape',done.shape ,'要求是[n,]')
        if done or truncated :
            print('environment done because','rewrd', reward, 'profit', profit,'step:',self.step_num)
            rmtree( self.creat_folder_n, ignore_errors = True)
        
        # print('in develop step check state shape', next_state.shape)
        return next_state, reward , agent_truncated,  agent_done , {'profit':profit} #必须要保证4个或者5个结果的输出（这个要求，不然会报错）
    
    # def _get_drawbridge_angle(self):
    #     return -min(max(self.current_step - self._drawbridge_start, 0.)*0.4, 90.0)
  
    def reset(self, seed=None, options={'ith': 0, 'record' : None}, test_model = None, del_well = None):#{new_path : None, 'n_new_well': None):#reset 就是运算第0步, add_well [n,2]
        print('begin to reset the environment')
        # if test_model is not None: #添加了新井的文件
        f_name = test_model['new_path'].split('/')[-1]
        folder_name = test_model['new_path'][:-(len(f_name)+1)] #也删去中间的/符号
        new_path = folder_name + test_model['copy_name']
        self.origin_folder = folder_name#原始文件所在文件夹
        self.creat_folder = new_path#新创建副本文件夹
        self.name = f_name
        self.n_new_well = test_model['n_new_well']

        self.well_num = self.args.n_old_well + self.n_new_well 
        # assert self.args.well_num == self.well_num == self.args.n_old_well + self.args.n_new_well, '井数对不上 {self.args.well_num} ，{self.args.n_old_well} ，{self.args.n_new_well}'
        self.acc_reward = np.array([0. for i in range(self.well_num)]).reshape(-1,1)
        if test_model['new_well_loc'] is None:
            low = np.ones(2)
            high = np.array(self.args.grid_dim[-2:])
            self.well_loc = np.stack([np.random.randint(low,high) for i in range(self.n_new_well)],axis=0) #[n_new, 2] 注意是y,x
        else:
            self.well_loc = test_model['new_well_loc']
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=( self.well_num, 6), dtype=np.float)
        if self.n_new_well>0:
            self.comp_grid = self.perf_2d_2_3d( self.well_loc, perf_start=0, perf_end = 6)#新井的射孔位置
        else:#如果处于删除井的状态
            self.comp_grid = np.array([self.n_new_well])
        # else:

        #     self.n_new_well = 0
        #     self.well_loc = np.array([[]])
        #     self.well_num = self.args.n_old_well + self.n_new_well 
        #     self.observation_space = spaces.Box(
        #         low=-1, high=1, shape=( self.well_num, 3, self.args.grid_dim[0], self.args.well_range*2, self.args.well_range*2,), dtype=np.float)
        #     self.acc_reward = np.array([0. for i in range(self.well_num)]).reshape(-1,1)
        #     self.comp_grid = self.perf_2d_2_3d( self.well_loc, perf_start=0, perf_end = 6)#新井的射孔位置
        # print('in env check well nu================ ', self.well_num)
            
        self.old_well_loc = pd.read_excel(self.args.posi_path).values#[n,4] #第一列序号，第二第三列是y,x 坐标 注意！！是y, x 坐标
        self.step_num = self.args.init_step
        if options['record'] is not None:
            self.record = options['record']
        else:
            self.record = None
        #随机井位
        if seed is not None:
            np.random.seed(seed)
        print(f'evn reset and determined well position is {self.well_loc}')
#重置文件
        # print('in develop rest file number is',options['ith'])
        self.creat_folder_n =  self.creat_folder + str(options['ith'])
        # print('in develop rest file path',self.creat_folder_n)
        self.path = self.creat_folder_n + '/' + self.name
        # print('in develop rest file name is',self.path)
        if os.path.exists(self.creat_folder_n):
            rmtree(self.creat_folder_n, ignore_errors=True)#结束后删掉复制的运算文件
        copytree(self.origin_folder, self.creat_folder_n)
        # print('in develop rest final  file name is',self.creat_folder_n,'++++++++++\n', self.path)
        #开始重置环境
        # self.step_num = self.args.init_step#得到初始的步数
        # self.well_list = None
        # print('++++++++++++++++++++++++++++++++++++++')

        run_path = write_and_run(self.args, self.path, self.well_list,  None, self.comp_grid, self.step_num, 1, self.args.init_step, self.record) 
        
        # print('****** run path in reset', run_path)
        self.current_path = run_path
        # print('in develop check reset',run_path)
        # print('----------------------------------------')
        self.indice = check_indice(run_path + '.RSM', time_len=1, key_words = self.key_list_total,restart = True)
        #step为0时，所有的参数都失效(包括action），只会运算源文件里面
        # sat = self.get_obs(run_path)[:,0,...]#n,z,r,r

        print('environment reset success')
        # reward = self.get_reward()
        # self.his_a_ob = sat
        # return self.get_obs(), self.get_state()
        return self.get_state(), {}
    
    def agg_adj(self, thrd=0.105 ,self_percent = 0.5):#path, [n,d], number of nerighbor, 筛选邻井的阈值, 自身占比
        a = self.total_well_loc
        adj = np.ones((a.shape[0],a.shape[0]))
        eps = 0.00001
        for i in range(a.shape[0]):
            for j in range(a.shape[0]):
                adj[i,j] = np.sqrt(((a[i] - a[j] )**2).sum())
        #继续处理
        # adj_log = np.log(adj+eps)
        adj_log = adj
        # print(adj_log)
        adj_inv = (adj_log.max() - adj_log)
        #让自身取0
        adj_inv = adj_inv - np.diag(np.diag(adj_inv))
        # print(adj_inv)
        # adj_norm = (adj_inv - adj_inv.min())/ (adj_inv.max() - adj_inv.min())
        pick_adj = adj_inv / adj_inv.sum(axis = 0, keepdims = True) #这里softmax的轴没问题，就是n那个轴，目的是知道当前行的某一口井为当前行提供了多大的占比
        final_adj = pick_adj.copy()
        final_adj[pick_adj< thrd] = 0
        final_adj = final_adj / (final_adj.sum(axis = -1, keepdims = True) + eps)   #
        final_adj += np.diag([self_percent for i in range(final_adj.shape[0])]) #[n,n]
        #整体思路就是让周围井也发挥一半作用，但是邻井被分担的越多，发挥作用越小
        # ind = np.argsort(adj_norm)[:,:n]
        return final_adj #b
        
    def get_avail_agent_actions(self, well_id):#判断这四个类型是不是都能随意转换
        return np.array([1 for i in range(self.args.type_dim)])#就是全都可以转换

    def perf_2d_2_3d(self, position, perf_start=0, perf_end = 6):#[n,2], 射孔顶层，射孔低层 [y,x]
        if position.shape[-1] == 0:
            return np.array([0])
        else:
            if perf_end == -1:
                perf_end = self.args.grid_dim[0]
            # z_num = perf_end - perf_start
            # perf_grid = np.zeros((position.shape[0],position.shape[1],z_num))
            well_p = []
            for line in range(position.shape[0]):
                well_single = []
                for j in range(perf_start, perf_end,1):
                    well_single.append([j+1, position[line,0],position[line,1]])
                single = np.array(well_single)#[g,3]
                well_p.append(single)        
            final = np.stack(well_p,axis = 0)#n,g,3
            return final.astype(int)
            
    
    def cal_xyz(self, init_grid_x, init_grid_y, inp_l = 300,dev = 0.2,azi = 0.6,grid_dim = [20,20,10],delt_grid = [10,50,50]):
    
        #输入信息
        init_grid = [1, init_grid_y,init_grid_x] #z,y,x
        
        #计算
        delt_z = np.cos(dev*np.pi) * inp_l
        delt_xy = np.sin(dev*np.pi) * inp_l
        delt_y = delt_xy * np.cos(azi * np.pi)
        delt_x = delt_xy * np.sin(azi * np.pi)
        delt_xn, delt_yn, delt_zn = np.ceil(delt_x/delt_grid[-1]), np.ceil(delt_y/delt_grid[-2]), np.ceil(delt_z/delt_grid[-2])
        
        n_max = int(max(delt_xn, delt_yn, delt_zn))
        dx_coor, dy_coor, dz_coor = [], [], []
        for i in range(n_max):
            dx_coor.append(np.ceil((delt_xn/n_max) * i))
            dy_coor.append(np.ceil((delt_yn/n_max) * i))
            dz_coor.append(np.ceil((delt_zn/n_max) * i))
        
        x_coor , y_coor, z_coor = np.array(dx_coor) + init_grid[-1], np.array(dy_coor) + init_grid[-2], np.array(dz_coor) + init_grid[-3]
        return x_coor , y_coor, z_coor
    
    
    def perf_grid(self, well_info, grid_dim = [20,20,10],delt_grid = [10,50,50], buffer_size = 10):#变量维度是cal_xyz基础上前面加上一个n维度
        init_grid_x, init_grid_y, inp_l ,dev ,azi = well_info[:,0],well_info[:,1],well_info[:,2],well_info[:,3],well_info[:,4]
    #生成矩阵【n,n_grid,3]
    #buffer就是最多存储10个点
        num_well = inp_l.shape[0]
        total = np.zeros((num_well, buffer_size,3))
        amin = np.ones((num_well, buffer_size,3))
        amax = np.stack([np.ones((num_well, buffer_size))*grid_dim[-1],np.ones((num_well, buffer_size))*grid_dim[-2],np.ones((num_well, buffer_size))*grid_dim[-3]], axis = -1)
        # print('in perf grid+++++++++++ check min  and max', amin,amax)
        for n in range(num_well):
            xset, yset, zset = self.cal_xyz(init_grid_x[n], init_grid_y[n], inp_l[n] ,dev[n] ,azi[n], grid_dim, delt_grid)
            for j in range(xset.shape[0]):
                total[n,j,0] = xset[j]
                total[n,j,1] = yset[j]
                total[n,j,2] = zset[j]
        #超出的网格设定为无效网格
        total[total<amin] = 0
        total[total>amax] = 0
        # clip_total = np.clip(total,amin, amax) 不是用这个函数，他会产生轨迹以外的网格
        return total.astype(int)
    
    def read_connect(self, well_loc, i,j, sat_field):
        if i == j :
            return 0
        else:
            delta = well_loc[j] - well_loc[i]
            path_ind = []
            for n in range(abs(delta).max()):
                # print('+++++++++++++++++++++++',well_loc.shape,i,j)#well_loc[j], well_loc[i],delta )
                grid = well_loc[i] + (delta//(abs(delta).max()))*n
                path_ind.append(grid)

            p_ind_array = tuple([np.array(path_ind)[:,0],np.array(path_ind)[:,1]])
            sat = sat_field[p_ind_array].mean()
            return sat
    
    def adj_state(self, well_loc, indice, state_obs, well_range):# [n,2],[n,d],[c,y,x]包括了邻接井点的【sat，pres，orintation，distance，acc inj, acc oil,]
        assert len(indice.shape) ==2, f'indice shape is {indice.shape}'
        ind_a = (well_loc -1)
        oil_ind = 0
        water_ind = -1
        inj_ind = 2
        bhp_ind = -2
        liquid_ind = 1
        sat = np.zeros((ind_a.shape[0],ind_a.shape[0]))
        pres = np.zeros((ind_a.shape[0],ind_a.shape[0]))
        orintation = np.zeros((ind_a.shape[0],ind_a.shape[0]))
        distance = np.zeros((ind_a.shape[0],ind_a.shape[0]))
        oil = np.zeros((ind_a.shape[0],ind_a.shape[0]))
        inj = np.zeros((ind_a.shape[0],ind_a.shape[0]))
        water = np.zeros((ind_a.shape[0],ind_a.shape[0]))
        delta_p = np.zeros((ind_a.shape[0],ind_a.shape[0]))
        o_ct = np.zeros((ind_a.shape[0],ind_a.shape[0]))#产油占比，不用含水而用含油的原因是注水井含水计算会是0含油计算也是0
        inter = np.zeros((ind_a.shape[0],ind_a.shape[0]))
        for i in range(ind_a.shape[0]):
            for j in range(ind_a.shape[0]):
                # sat[i,j] = state[0,ind_a[j,0],ind_a[j,1]]
                # pres[i,j] = state[1,ind_a[j,0],ind_a[j,1]]
                distance[i,j] = np.sqrt(((ind_a[i] - ind_a[j])**2).sum())
                orintation[i,j] = np.arctan2(ind_a[i,0] - ind_a[j,0], ind_a[i,1] - ind_a[j,1])
                # inj[i,j] = indice[j,inj_ind]
                # oil[i,j] = indice[j,oil_ind]
                # water[i,j] = indice[j,water_ind]
                # delta_p[i,j] = indice[i,bhp_ind]- indice[j,bhp_ind]
                o_ct[i,j] = indice[j,oil_ind]/(indice[j,water_ind]+ indice[j,oil_ind]+0.01)
                inter[i,j]=self.read_connect(ind_a, i,j,state_obs[0])
        
        # state = np.stack([sat,pres,orintation,distance, inj, oil, water],axis = -1)#[n,n,d]
        state = np.stack([orintation,distance,o_ct, inter],axis = -1)#[n,n,d]
        # print(state.shape,'-------------------------------------',well_range,ind_a.shape)
        
        #采用井数限制方式来卡
        pick_ind = np.argsort(np.repeat(state[:,:,0:1],state.shape[-1],axis = -1), axis=1)[:, :well_range,:]#找出最小的两个的坐标[n,n,d]
        limit_state = np.take_along_axis(state, pick_ind, axis=1)#[n,well_range,d]
        # #采用距离限制方式来卡
        # #找到需要聚合的点
        # delta_p[delta_p<=0]=0
        # adj = delta_p/(distance+0.01)
        # # adj[distance>=well_range]==0
        # # adj_max = adj.max(-1,keepdims = True)
        # # adj[distance<well_range]==(adj_max - adj)//adj_max
        # adj = adj / (adj.sum(0,keepdims = True)+0.01) + np.eye(adj.shape[0])
        # adj = adj.reshape(*adj.shape,1)
        # limit_state = state
        # print(limit_state.shape,'===========================================')
        #开始norm
        # min_list = [self.args.sat_min, self.args.pres_min, -3.142,0,0,0,0]
        # max_list = [self.args.sat_max, self.args.pres_max, 3.142, max(self.args.grid_dim[-2:]), 100,100,100]
        min_list = [-3.142,0,0,0]
        max_list = [3.142, max(self.args.grid_dim[-2:]), 1,1]
        for s in range(limit_state.shape[-1]):
            limit_state[...,s] = self.norm(limit_state[...,s], max_list[s], min_list[s])
        final_state = limit_state.reshape(limit_state.shape[0],-1) #* adj
        
        return final_state#.mean(1)#n,d
    
    def agg_adj_reward(self, well_loc, indice,dis_thrd):# [n,2],[n,d],[c,y,x]包括了邻接井点的【sat，pres，orintation，distance，acc inj, acc oil,]
        assert len(indice.shape) ==2, f'indice shape is {indice.shape}'
        ind_a = (well_loc -1)
        oil_ind = 0
        inj_ind = 2
        bhp_ind = 4
        liquid_ind = 1
        # sat = np.zeros((ind_a.shape[0],ind_a.shape[0]))
        # pres = np.zeros((ind_a.shape[0],ind_a.shape[0]))
        # orintation = np.zeros((ind_a.shape[0],ind_a.shape[0]))
        distance = np.zeros((ind_a.shape[0],ind_a.shape[0]))
        acc_inj = np.zeros((ind_a.shape[0],ind_a.shape[0]))
        acc_pro = np.zeros((ind_a.shape[0],ind_a.shape[0]))
        bhp = np.zeros((ind_a.shape[0],ind_a.shape[0]))
        delta_p = np.zeros((ind_a.shape[0],ind_a.shape[0]))
        liquid = np.zeros((ind_a.shape[0],ind_a.shape[0]))
        
        for i in range(ind_a.shape[0]):
            for j in range(ind_a.shape[0]):

                # sat[i,j] = state[0,ind_a[j,0],ind_a[j,1]]
                # pres[i,j] = state[1,ind_a[j,0],ind_a[j,1]]
                distance[i,j] = np.sqrt(((ind_a[i] - ind_a[j])**2).sum())
                # orintation[i,j] = np.arctan2(ind_a[i,0] - ind_a[j,0], ind_a[i,1] - ind_a[j,1])
                acc_inj[i,j] = indice[j,inj_ind]
                acc_pro[i,j] = indice[j,oil_ind]
                bhp[i,j] = indice[j,bhp_ind]
                #本来是希望通过bhp建立供应关系，但是高渗透油藏来说，整个油藏压力基本没有很明显压力梯度，因此改用液量
                delta_p[i,j] = (indice[i,inj_ind] + indice[i,liquid_ind]*-1) - (indice[j,inj_ind] + indice[j,liquid_ind]*-1)#默认dim=0维度是源头井，dim=1默认是尽头井

        #压差矩阵先过滤，删掉那些负值，就是源头井压力小于尽头井压力的
        delta_p[delta_p<=0]=0
        
        #找到需要聚合的点
        adj = delta_p/(distance**2+0.01)

        #开始便准话处理每一行，让总量保持1，也就是将压差变成分配系数
        adj = adj / (adj.sum(0,keepdims = True)+0.01) + np.eye(adj.shape[0])
        # print('in develop check adj++++++++++', delta_p,'+++++++++++++++\n',distance)
        state = np.stack([acc_inj, acc_pro],axis = -1)#[n,n,d]
        #开始过滤所有属性
        state_fil = state.copy()
        for i in range(state_fil.shape[-1]):
            state_fil[...,i] = state[...,i] * adj
        #开始聚合，并使用三种聚合方式
        # state_mean = state_fil.mean(-1)
        state_sum = state_fil.mean(1)#n,d
        # state_max = state_fil.max(-1)
        reward = ((state_sum[:,1]*1300-state_sum[:,0]*15)/10000) * ((self.args.max_steps - self.step_num)/self.args.max_steps)
        return reward.reshape(self.well_num,1) #n,well_range,d
    
    def close(self):
        return
    
    # def inv_position(self, well_posi):#[n]
    #     #还原尺度
    #     position = np.zeros(well_posi.shape)
    #     position[:,0] = well_posi[:,0] * (self.args.grid_dim[-1] - 1) + 1
    #     position[:,1] = well_posi[:,1] * (self.args.grid_dim[-2] - 1) + 1
    #     position = position.astype(int)
    #     return position
    
    # def norm_position(self, well_posi):#[n]
    #     #标准化尺度
    #     position = np.zeros(well_posi.shape)
    #     position[:,0] = (well_posi[:,0]-1) / (self.args.grid_dim[-1] - 1) 
    #     position[:,1] = (well_posi[:,1]-1) / (self.args.grid_dim[-2] - 1) 
    #     # position = position.astype(int)
    #     return position
    def well_posi_map(self, well_posi):#n,g,3 包括老井 让开的井取1，关的井取0
    #定义场数据
    
        proj_perf = well_posi.reshape(-1,3)
        print('------------------- in develop check compgird',proj_perf.shape,proj_perf)
        square = np.zeros((self.args.grid_dim[0],self.args.grid_dim[1],self.args.grid_dim[2]))
        for num, w in enumerate(proj_perf):
            # print(w)
            square[w[0]-1, w[1]-1, w[2]-1] = 1 #需要比实际网格号小1
        return square
    def cluster(self, well_loc, grid_dim, actnum, distance_thrd=3, divide_layer = True):#[n,2] [z,y,x], [z,y,x]
       n_cluster = well_loc.shape[0] #n 个cluster
       # print('+++++++++++++++++1 ', n_cluster, well_loc)
       grids = np.ones(grid_dim[-2:])#[y,z]
       # grids[actnum[0,...]==0] = 0
       # sns.heatmap(grids)
       # plt.show()
       grid_x ,grid_y = np.where(grids == 1)
       
       index_grid = np.stack([grid_y.reshape(grid_dim[-2],grid_dim[-1]),grid_x.reshape(grid_dim[-2],grid_dim[-1])],axis = -1)#[y,x,2]#dim 是他的y和xindex
       coord = index_grid + 1 #序号要加1
       # sns.heatmap(coord[...,0])
       # plt.show()
       # sns.heatmap(coord[...,1])
       # plt.show()
       
       dist_list = []
       for w in range(n_cluster):
           distance = np.sqrt((coord[...,1]- well_loc[w,0])**2 + (coord[...,0]- well_loc[w,1])**2)#[y,x]
           dist_list.append(distance)
           # sns.heatmap(distance)
           # plt.show()
       dist_m = np.stack(dist_list,axis = 0)#n,y,x
       dist_m[dist_m>distance_thrd] = -1#np.inf #有重复的会重复筛选
       cluster_list = []
       for w in range(n_cluster):
           full_field = np.stack([dist_m[w] for i in range(grid_dim[0])],axis = 0)# z,y,x
           # sns.heatmap(dist_m[w][...])
           # plt.show()
           full_field[actnum==0] = -1

           # sns.heatmap(full_field[0,...])
           # plt.show()
           if divide_layer:
               up, bot = grid_dim[0]//2, grid_dim[0]//2
               full_field_up = full_field.copy()
               full_field_up[-bot:] = -1#np.inf
               full_field_bot = full_field.copy()
               full_field_bot[:up] = -1#np.inf

               part_up = np.where(full_field_up>=0 )
               part_bot = np.where(full_field_bot>=0 )
               cluster_list.append(part_up)
               cluster_list.append(part_bot)
           else:
               cluster_list.append(np.where(full_field>=0))#排除死网格的统计
           
       #for show 
       # sns.heatmap(actnum[0,...])
       # plt.show()
       # print('=============================', cluster_list[0])
       # show = np.zeros(grid_dim)
       # ind_s = 1
       # for ind in cluster_list:
       #     show[ind] = show[ind]+ ind_s
       #     ind_s += 1
       # sns.heatmap(show[0,...])
       # plt.show()
       # sns.heatmap(show[-1,...])
       # plt.show()
       return cluster_list # 长度是n 每一个elemet 是np.where 可以直接作为索引[z,y,x] 输出格式：第一口井上层，第一口井下层，第二口上，第二口下。。。。