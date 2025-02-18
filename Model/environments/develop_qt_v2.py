import os
print(os.path.abspath('.'))
import numpy as np
import gym
from gym import spaces
from gym import Env
from gym.utils import seeding
from environments.env_utils_qt import write_and_run,  check_field, check_indice, npv_fun , agent_npv, agg_adj

from shutil import copy, copytree, rmtree
import seaborn as sns
import pandas as pd
class development_qt2(Env):  
    #path 为不带后缀的母文件名
    def __init__(self, args = None,position = None):
        super().__init__()

        #创建一个副本，在副本上运算
        self.args = args
        f_name = args.path.split('/')[-1]
        folder_name = args.path[:-(len(f_name)+1)] #也删去中间的/符号
        new_path = folder_name + '_new_'
        # self.file = f_name #EGG_MODEL_ECL
        # self.folder = folder_name#G:/optimiztion_HRL/SAC_egg/egg/
        self.add_step_num = None
        self.well_style = args.well_style
        self.origin_folder = folder_name#原始文件所在文件夹
        self.creat_folder = new_path#新创建副本文件夹
        self.name = f_name
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
        self.acc_reward = np.array([0. for i in range(len(self.well_list))])
        # if args.well_style =='vertical':
        #     self.well_posi = position
        #     self.comp_grid = self.perf_2d_2_3d(position, perf_start=1, perf_end = 30) #[n_old,2]
        # elif args.well_style =='horizon':
        #     add_info = np.array([[200,0,0] for i in range(position.shape[0])])
        #     self.well_posi = np.concatenate([position,add_info],axis = 1)# 从n,2 变成n，5
        #     self.comp_grid = self.perf_grid(self.well_posi, grid_dim = args.grid_dim, delt_grid = args.grid_delta)
        self.args = args
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
            low=-1, high=1, shape=(3*args.well_num,), dtype=np.float)
            
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.args.action_dim,), dtype=np.float)
        assert (self.args.state_dim[0] == self.observation_space.shape[0] ), 'args 设定的state dim 与env的ob dim不一致'
        assert self.args.well_num == (self.args.n_old_well + self.args.n_new_well)
        assert self.args.action_dim == self.args.well_num,'action 维度不对'
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
        return (x - (max_value + min_value)/2) /((max_value - min_value)/2)
        

    def compute_reward(self, achieved_goal, desired_goal, info):
        #额外多出来的函数，直接等于已完成目标就可以了呀
        current = achieved_goal
        return current
    
    def get_reward(self):
        
#计算多部模拟的效益
        ob_keys = ['WOPR', 'WLPR', 'WWIR', 'WGIR','WBHP']
        actor_ob_list = []
        include_keys = []
        for i in range(len(ob_keys)):
            actor_data = self.indice[self.key_list_total.index(ob_keys[i])] #(1,n)
            if len(actor_data.shape)==1: actor_data = actor_data.reshape(1,actor_data.shape[0]) #只有一口井的时候，维度无法保持n那个维度，所以要强制加一维保持队形
            # print('in develop checck actor shape', actor_data.shape)
            if ob_keys[i] not in include_keys:
                actor_ob_list.append(actor_data)
                include_keys.append(ob_keys[i])

        well_indice = np.stack(actor_ob_list,axis = -1).squeeze(0)#(n,4)
        # print('check well indice shape in develop.py++++++++++',self.indice[0].shape, well_indice.shape, well_indice)
        profit = (well_indice[:,0] * self.args.oil_price * self.args.interval \
                       + well_indice[:,1] * self.args.w_price * self.args.interval \
                       + well_indice[:,2] * self.args.inj_w_price * self.args.interval \
                           + well_indice[:,3] * self.args.inj_g_price * self.args.interval)/1000000 #[n]
            
        well_type_all_oil = ((well_indice[:,1] - well_indice[:,2])>0).all()
        well_type_all_water = ((well_indice[:,1] - well_indice[:,2])<0).all()
        
        # 采用自定义的reward来让action学习
        
        # reward = (profit.sum()>0)*1 + well_type_all_oil*-1 + well_type_all_water*-1 + max(((self.args.pres_min+100) - self.indice[self.key_list_total.index('FPR')][0]),0)*-1  \
        #     + self.new_well_near * -2 #距离进了不行
        reward = profit.sum()
        # print('in develop env +++++++****************+', reward, profit)
        return reward, profit.sum()#[n,z,r,r,]        

    def get_state(self):#生成agent的记录观测量
        # print('chech obs run path',self.current_path)
    # 三维场
    #     run_path = self.current_path
    #     p_act = self.path + '.FEGRID'
    #     p_att = run_path + '.F'+ str(self.step_num+1).zfill(4) #因为第0步生成F0001是查询的目标
    #     sat, pres, _ = check_field(p_act, p_att, dim = self.dim, squeeze= True) #第0步生成0 和1 两个，读取的应该是1，维度是(25000,)
    # #标准化观测结果
    #     sat_ob = self.norm(sat, self.args.sat_max,self.args.sat_min)
    #     pres_ob = self.norm(pres, self.args.pres_max,self.args.pres_min)

    #     # 计算所有井的comp_grid

    #     # well_map = self.well_posi_map(self.comp_grid)
        
    #     step_map = np.ones((sat_ob.shape)) * (self.step_num / self.args.max_steps)
    #     # sns.heatmap(well_map[0,...])
    #     # plt.show()
    #     agent_state = np.stack([sat_ob, pres_ob, step_map],axis = 0)#[1,4,z,r,r]
    
        ob_keys = ['WWCT', 'WBP9']
        actor_ob_list = []
        include_keys = []
        for i in range(len(ob_keys)):
            actor_data = self.indice[self.key_list_total.index(ob_keys[i])] #(1,n)
            if len(actor_data.shape)==1: actor_data = actor_data.reshape(1,actor_data.shape[0]) #只有一口井的时候，维度无法保持n那个维度，所以要强制加一维保持队形
            # print('in develop checck actor shape', actor_data.shape)
            if ob_keys[i] not in include_keys:
                actor_ob_list.append(actor_data)
                include_keys.append(ob_keys[i])
        actor_ob_list.append(np.ones(actor_ob_list[0].shape) * self.step_num)
        well_indice = np.stack(actor_ob_list,axis = -1).squeeze(0)#(n,3)
        state = self.norm(well_indice, np.array([[0,self.args.pres_min,1]]), np.array([[1,self.args.pres_max,self.args.max_steps]])).flatten()
        assert state.shape[0] == self.args.state_dim[0], f'输出的state维度和设定不一致，输出{state.shape[0]},设定的是{self.args.state_dim}'
        print('in develop env check state',well_indice.shape,  state.shape)
        return state #n,2,z,r,r
        
    def step(self, action):# [n] #
        #因为每一步他要先都step之前的ob所以，step的升级必须再当前步
        self.step_num += 1
        assert action.shape[0] == self.args.well_num ,' 井数不对'
        
        run_path = write_and_run(self.args, self.path, self.well_list, action,  self.step_num, self.step_time, self.args.init_step, self.record)
        if self.step_num == 0: 
            restart = False
        else:#从第1步开始，我们虽然生成第1步时还是考虑使用的原文件，但要查看的结果已经是restart文件格式了
            restart = True


        copy(run_path+'.RSM', run_path+'_c.RSM')
        self.indice = check_indice(run_path + '.RSM', time_len=1, key_words = self.key_list_total,restart = restart)
        
        
        self.current_path = run_path
        reward, profit = self.get_reward()
        #计算全局收益

        # #done的判别收益
        self.acc_reward += reward
        
        next_state = self.get_state()

        # print('in develop check step num+++++++++++++++++', self.step_num)

        #终止条件 
        #奖励达到了，或者最大步数超过了，或者不再盈利了
        # print('in develop env check npv', 'reward:',reward.shape,'judge reward is ', delt_a_ob.shape)
        done = True if (self.step_num - self.args.init_step)>= self.args.max_steps  or reward <=0 else False#g_reward < 0 #+ np.array([self.step_num >= self.max_episode_length for i in range(action.shape[0])]) #or partial_obs['profit'] <= 0
        truncated = True if reward <=0 else False
        # print('in develop check done shape',done.shape ,'要求是[n,]')
        if done or truncated :
            print('environment done because','rewrd', reward,'step:',self.step_num)
            rmtree( self.creat_folder_n, ignore_errors = True)
        
        # print('in develop step check state shape', next_state.shape)
        return next_state, reward , truncated, done, {'real_r':profit} #必须要保证4个或者5个结果的输出（这个要求，不然会报错）

    # def _get_drawbridge_angle(self):
    #     return -min(max(self.current_step - self._drawbridge_start, 0.)*0.4, 90.0)
  
    def reset(self, seed=None,options={'ith': 0, 'record' : None}):#reset 就是运算第0步, add_well [n,2]
        print('begin to reset the environment')
        
        self.step_num = self.args.init_step
        if options['record'] is not None:
            self.record = options['record']

        else:
            self.record = None
#重置文件
        print('in develop rest file number is',options['ith'])
        self.creat_folder_n =  self.creat_folder + str(options['ith'])
        print('in develop rest file path',self.creat_folder_n)
        self.path = self.creat_folder_n + '/' + self.name
        print('in develop rest file name is',self.path)
        if os.path.exists(self.creat_folder_n):
            rmtree(self.creat_folder_n, ignore_errors=True)#结束后删掉复制的运算文件
        copytree(self.origin_folder, self.creat_folder_n)
        # print('in develop rest final  file name is',self.creat_folder_n,'++++++++++\n', self.path)
        #开始重置环境
        # self.step_num = self.args.init_step#得到初始的步数
        # self.well_list = None

        run_path = write_and_run(self.args, self.path, self.well_list, None, self.step_num, 1, self.args.init_step, self.record) 
        
        # print('****** run path in reset', run_path)
        self.current_path = run_path
        print('in develop check reset',run_path)
        self.indice = check_indice(run_path + '.RSM', time_len=1, key_words = self.key_list_total,restart = True)
        #step为0时，所有的参数都失效(包括action），只会运算源文件里面
        # sat = self.get_obs(run_path)[:,0,...]#n,z,r,r

        print('environment reset success')
        # reward = self.get_reward()
        # self.his_a_ob = sat
        # return self.get_obs(), self.get_state()
        return self.get_state(), {}

    
    def get_avail_agent_actions(self, well_id):#判断这四个类型是不是都能随意转换
        return np.array([1 for i in range(self.args.type_dim)])#就是全都可以转换

    def perf_2d_2_3d(self, position, perf_start=0, perf_end = 6):#[n,2], 射孔顶层，射孔低层 [y,x]
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
        
    def close(self):
        return
    

    def well_posi_map(self, well_posi):#n,g,3 包括老井 让开的井取1，关的井取0

    #定义场数据
        proj_perf = well_posi.reshape(-1,3)
        square = np.zeros((self.args.grid_dim[0],self.args.grid_dim[1],self.args.grid_dim[2]))
        for num, w in enumerate(proj_perf):
            # print(w)
            square[w[2]-1, w[1]-1, w[0]-1] = 1 #需要比实际网格号小1
        return square