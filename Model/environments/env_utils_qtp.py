import pandas as pd
import math
import torch
import numpy as np
from subprocess import run

import os
       

def write_and_run (args, path, well_list, quantities, comp_grid, step_num, interval, init_step = None,record_file = None):#path =/egg/EGG_MODEL_ECL , [ pro/inj, type]
    # if comp_grid is None:
    #     comp_grid = np.zeros((0,3,2))
    # print('++++++++++++++++++++++++++++++++++++++++++++',(comp_grid == np.zeros((3,3,3))).all())
    # print('in env util check open time',open_time_ind)
    # new_open_time = [open_time_ind[:i+1].sum() for i in range(len(open_time_ind))] #从开井的间隔时常变成实际开始时间点, 
    # open_time  = [0 for i in range(args.n_old_well)] + list(new_open_time)#[n] open time 要包含所有井 ，old在前，new在后
    
    case_name = path.split('/')[-1]
    case_path = path[:-len(case_name)-1]
    print('case path is ',path)
    #开始
    if step_num == 0:#
        # print('lllllllllllllllllllllll run file',path,step_num)
        casedat = path + '.DATA'
        command = 'eclrun eclipse' + ' ' + casedat
        # os.system(command)
        run(command, shell=True)
        # print('env util check run path', casedat)
        
    else:
        if well_list is None:
            well_list = ['INJECT1','INJECT3', 'PROD1' ]
        case_name = path.split('/')[-1]
        # inj_list = []#所处位置
        # pro_list = []
        # inj_value = []
        # pro_value = []
        # shut_list = []
        #定义新井的名字
        new_well_list = ['W1','W2','W3','W4']
        # print('in env util,check well type',type_code)
        #注入类型
        
        #还原变量尺度 #n,1 包含了固新加的井位 x*delta + mid
        assert args.control_min <0 , '注采量+井别的设置必须保证量同时存在正负，请修改control min 的值'
        
        if quantities is None:
            quantities = np.array([0 for i in range(args.well_num)]).astype(int)
        else:
            quantities = quantities * ((args.control_max - args.control_min)/2) + ((args.control_max + args.control_min)/2)
            quantities = quantities.astype(int)
        well_list_total = well_list + new_well_list[:comp_grid.shape[0]] #保持变化的经数量
        assert len(well_list_total) == args.well_num, f'井列表数量不足,总井列表{well_list_total},{comp_grid.shape[0]}'
        assert quantities.shape[0] == len(well_list_total), f'quantities 数量和井数不匹配，quantities数量{quantities.shape[0]}，井数{len(well_list_total)}'
        pro_list = []
        pro_value_list=[]
        inj_list = []
        inj_value_list=[]

        for wt in range(len(well_list_total)):
            # if abs(quantities[w_t])< 10:
            #     well_type[w_t] = 0
            if quantities[wt]>=0 :
                pro_list.append(well_list_total[wt])
                pro_value_list.append(quantities[wt])

            elif quantities[wt]<0 :
                inj_list.append(well_list_total[wt])
                inj_value_list.append(quantities[wt] * -1)

        print('本次选择的控制量是',quantities)
        # if record_file is not None:
        #     record_file.write('目前开井数:')
        #     record_file.write(f'{len(well_list_total)}\n')
        #     record_file.write('本次选择的井位置是:')
        #     record_file.write(f'{comp_grid[:,0,:2]}\n')
        #     record_file.write('本次选择的控制量是:')
        #     record_file.write(f'{quantities}\n')
        #     record_file.flush()
        #开始编辑运算的重启状态文件solution部分
        sol_name = case_path +'/'+ f'RESTART{step_num}_INIT.INC'
        sol = open(sol_name, 'w', encoding='utf-8')
        sol.write('RESTART\n')
        if step_num == init_step :
            sol.write(f'{case_name} {step_num} / \n')
        else:
            sol.write(f'RESTART{step_num-1} {step_num} 1* 1* / \n')
        sol.close()
        #开始编辑运算SCH文件
        sch_name = case_path +'/'+  f'RESTART{step_num}_SCH.INC'
        # print('*********************', sch_name)
        file = open(sch_name, 'w', encoding='utf-8')
        file.write('RPTRST\n')
        file.write('BASIC=2 / \n')
        file.write('\n')

        #吧新井的定义和射孔定义出来

        file.write('WELSPECS\n')
        for num in range(comp_grid.shape[0]):
            file.write(f'{new_well_list[num]} 1 {comp_grid[num,0,-1]} {comp_grid[num,0,-2]} 1* OIL 9* STD / \n')
        file.write('/\n')
        file.write('COMPDAT\n')
        for num in range(comp_grid.shape[0]):
            for grid in range(comp_grid[num].shape[0]):
                if (comp_grid[num][grid]==0).any(): 
                    continue
                else:
                    file.write(f'{new_well_list[num]} {comp_grid[num,grid,-1]} {comp_grid[num,grid,-2]} {comp_grid[num,grid,-3]} {comp_grid[num,grid,-3]}  OPEN 2* 0.2 1* 0 3* / \n')
        file.write('/\n')
            
        file.write('WCONINJE\n')
        for num, name in enumerate(inj_list):
            file.write(f'{name}   WATER    OPEN    RESV    1*  {inj_value_list[num]}    600 /  \n')#[0]是为了去掉结果中的[]符号
        file.write('/ \n')
        file.write('\n')
        file.write('WCONPROD\n')
        for num, name in enumerate(pro_list):
            file.write(f'{name} 1* RESV 4* {pro_value_list[num]} 300 / \n')
        file.write('/\n')


        file.write('\n')
        file.write('TSTEP\n')
        file.write(f' {interval} / \n')
        file.write('\n')
        #file.write('TUNING\n')
        #file.write('0.1 30 / \n')
        #file.write('/ \n')
        #file.write('12 1 250 1* 25 / \n')
        #file.write('\n')
        file.close()
#开始编辑最原始的data文件
        old_data = path +'.DATA'
        # print('++++++++++++++++++++++++', old_data)
        with open(old_data,'r', encoding='utf-8') as old_data_f:
            old_data_lines = old_data_f.readlines()
            for keynum, line in enumerate(old_data_lines):
                if 'SCHEDULE' in line:
                    sch_posi = keynum+3
                elif 'SOLUTION' in line:
                    sol_posi = keynum+3
        # print(old_data_lines, sch_posi, sol_posi)
        data_name = case_path+ '/'+ f'RESTART{step_num}.DATA'
        # print('++++++++++++++', data_name)
        data_file = open(data_name, 'w+', encoding='utf-8')
        data_file.writelines(old_data_lines[:sol_posi])
        data_file.write(f'\'RESTART{step_num}_INIT.INC\'  /\n')
        data_file.writelines(old_data_lines[sol_posi+1:sch_posi])
        data_file.write(f'\'RESTART{step_num}_SCH.INC\'  /\n')
        data_file.writelines(old_data_lines[sch_posi+1:])
        data_file.close()
        
#输出data文件路径
        casedat = data_name
        command = 'eclrun eclipse' + ' ' + casedat
        # os.system(command)
        run(command, shell=True)
        
    return '.'.join(casedat.split('.')[:-1]) #不要后缀  使用.符号的分隔符来重新融合除了后缀以外的所有字符，这样可以防止后缀前面还有.



def check_indice(p_dir='g:/python/代理模型/test',time_len=10, key_words = ['WLPR','WWIR'],restart = True):
    filename =f'{p_dir}'  #需要转化的文件
     
    #显示所有行
    pd.set_option('display.max_rows', None)
    
    case_name = filename.split('/')[-1].split('.')[0] if '/' in filename else filename.split('\\')[-1].split('.')[0]#找出最后一部分字符，也就是文件名
    #末尾添加标识
    with open(filename, encoding="utf-8",mode="a") as file:  
        file.write(f'\n\tSUMMARY OF RUN {case_name}') 
        file.close()
    
    # print('read file path is ', filename)
    df_t=pd.read_csv(filename,sep = '\t+',header= 1,engine='python')

    df_t.rename(columns=lambda x: x.strip(),inplace = True)#去掉列标题里面的前后空格
    # print(df_t)
    # df_t.to_excel('D:/test/tt2.xlsx')
    label = (df_t[df_t['TIME']==f'SUMMARY OF RUN {case_name}'].index+1).tolist()#关键字的提取，用于标记除了首行数据以外的其他行,最后一行是没有关键字的
    mark = (df_t[df_t['TIME']==f'SUMMARY OF RUN {case_name}'].index-time_len).tolist()#具体数值的提取，找出除了第一个标题以后的所有标题位置，便于拆分
    # mark.insert(0,2)#加上最初的位置
    
    # start=df_t[df_t.iloc[:,0]=='0'].index.tolist()#resa
    # print('...........................',mark)#数据开始
    # print('...........................',label)#关键字开始
    # print(start)#SUMMARY OF RUN 16LAYER                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
    df_list=[]
    
    # 注意，如果是开始时间点，那么必然包含全0的那一行，也就是会 比时间步多一行，
    #但是！ 如果是重启的，就没有最初哪一行
    # print('3333333333333333333333333', label,start)
    for i in range(len(label)):
        if i == 0 : #不使用label因为第一行的关键字是整个excel的列标题
            df_split = df_t.iloc[mark[i]:mark[i]+time_len,:].copy()
            df_split.columns=df_t.columns
            df_split.rename(columns =lambda x: x[0:5].strip(),inplace = True )# ！！感觉自己好强！
            
            
        elif i>0:#
            df_split = df_t.iloc[mark[i]:mark[i]+time_len,:].copy()
            # print(df_split,'????????????')
            df_split.columns=df_t.iloc[label[i-1],:].copy()#因为label比mark少一行，缺少了首行，因为首行的被作为整个pandas的columns了
            # print('lllllllll',df_split,'????????????')
        df_split.dropna(axis=1,how="all",inplace=True)
        df_split.rename(columns=lambda x: x.strip(),inplace = True)
    
        df_split = df_split.astype('float',errors='raise')  
        
        # print('once')
        # print(df_split,df_split.shape, '+++++++++++++++++++++++++++')
        df_split.set_index(['TIME'] ,drop=True , inplace = True)#重新指定index
        # print(df_split)
        # print(df_split.index,df_split)
        df_list.append(df_split)
    # print(df_list[-2])
    #列表数据横向合并
    # print(df_list)
    df_end=pd.concat(df_list,axis=1,copy=True)
    
    #df_end 就是全部的属性表了
    #下面分属性调用
    result = []
    for key in key_words:
        out = df_end[key].values
        result.append(out)
    # print('输出维度查询',len(result),result[0].shape,result[0])
    return result #F开头的维度是（1，）， W开头维度是（1，n）



def check_field(p_act, p_att, dim=[15000,2,1],squeeze= True):#actum 文件位置；其余attribute位置；维度； squeeze 用于只调取一个样本的情况，结果将返回一个场的尺度
    path_act = f"{p_act}" #actnum 所在的文件
    path_att = f"{p_att}" # 重启文件的场数据所在文件
    #dim=[z,y,z]
    sat_position = []#饱和度属性的位置
    pre_position = []
    sat_list = []#保存每一步的值
    pre_list = []
    
# 读取actnum数据
    total=[]
    with open(path_act) as f:
        lines=f.readlines()
        # print(lines)
        start=None
        for num in range(len(lines)):
            if lines[num].find('ACTNUM')!=-1:#-1表示没找到
                start = num+1
            
        for line in lines[start:]:
            if  '\'' in line : #进入下一个关键字
                break  
            else:
                for i in line.strip().split():
                    # print(i)
                    if '*' in i:#有*的就拆开
                        part=int(i.split('*')[0])*[int(i.split('*')[1])]
                        total += part
                        # print(total)
                    elif '/' in i :
                        break     #结束当前for循环，不结束更高一级循环      
                    else:
                        # print(i)
                        total += [int(i)]  
        # print('actnum 的数量是：',len(total))
    actnum = np.asarray(total).reshape(dim)  
    
#开始读取其他attribute
    #定位其他关键字
    with open(path_att) as f:
        lines=f.readlines()
        # print('read_dynamic begin to positon')
        for num in range(len(lines)):
            if lines[num].find('SWAT')!=-1:
                sat_position.append(num+1)
            if lines[num].find('PRESSURE')!=-1:
                pre_position.append(num+1)
        #开始计算    
        # print('read_dynamic begin to cal')  
        for s in sat_position:
            single_s=[]#每一个时间步的饱和度场，（只显示活网格）
            single_s_act = actnum.copy().astype(float)#有个问题，就是这个single_s_act 是直接完整调用了actnum的内存位置，后面一旦修改之后相当于actnum也修改了,所以用copy（）
            for line in lines[s:]:
                # print(line,s)
                if '\'' in line :
                    break
                else:
                    for i in line.strip().split():
                        single_s += [float(i)]#[1-float(i)]#列表形式 输出含油饱和度
            single_s_act[np.where(single_s_act==1)] = single_s#改正以后，没有0值了
            sat_list.append(np.expand_dims(single_s_act.reshape(dim),axis=0))
        # print('read_dynamic begin to recal')           
        for p in pre_position:
            single_p=[]#每一个时间步的饱和度场，（只显示活网格
            single_p_act = actnum.copy().astype(float)
            for line in lines[p:]:
                # print(line,s)
                if '\'' in line :
                    break
                else:
                    for i in line.strip().split():
                        single_p += [float(i)]#列表形式
            single_p_act[np.where(single_p_act==1)]=single_p
            pre_list.append(np.expand_dims(single_p_act.reshape(dim),axis=0))
    # print('begin to cat data')
    final_sat = np.concatenate(sat_list,axis=0)
    final_pre = np.concatenate(pre_list,axis=0)
    if squeeze == True:
        saturation, pressure, actnum = final_sat.squeeze(0), final_pre.squeeze(0), actnum

    else:
        
        saturation = np.expand_dims(final_sat,axis=0)
        pressure = np.expand_dims(final_pre,axis=0)
    # print('最终的尺寸是：[B,T，z,y,x]',saturation.shape,pressure.shape)
    # 合并
    # out = np.stack([saturation, pressure], axis = 0)
    # print('final field shape is ', saturation.shape)
    return saturation, pressure, actnum#j结果要拉平


def npv_fun(indice, indice_list = ['FOPR','FWPR','FWIR'],step_time = 180, step_num=1, price = None):
    fopr = indice[indice_list.index('FOPR')]#（1，）
    fwpr = indice[indice_list.index('FWPR')]
    fwir = indice[indice_list.index('FWIR')]
    fgir = indice[indice_list.index('FGIR')]
    
    fopt = indice[indice_list.index('FOPT')]#（1，）
    fwpt = indice[indice_list.index('FWPT')]
    fwit = indice[indice_list.index('FWIT')]
    fgit = indice[indice_list.index('FGIT')]
    # print('in env utils,llllllllllllllllllll',fopt,fwpt,fwit)
    if price:
        oil = price['oil']
        water = price['water']
        inj_w = price['inj_water']
        inj_g = price['inj_gas']
    else:
        oil = 1200
        water = -20
        inj_w = -30
        inj_g = -100
    # print('check output indices in env utils','fopt:',fopt,'fwpt',fwpt,'fwit', fwit, 'fgit','npv',fgit ,fopt * oil + fwpt * water + fwit * inj_w + fgit * inj_g)
    
    return (fopr * step_time * oil + fwpr * step_time * water + fwir * step_time * inj_w + fgir * step_time * inj_g)/10000 * (1/(1+0.12)**((step_num*step_time)/360)),\
            (fopt * oil + fwpt * water + fwit * inj_w + fgit * inj_g)/10000 #变成万元

def agent_npv(agent_ob, step_time = 180, step_num=1, price = None): # agent_ob [n,b,4]
    # 于上面npv fun的区别就是上面计算总的油藏收益，下面计算每个agent 也就是每口井的收益
    wopr = agent_ob[:,0]#(n,)
    wwpr = agent_ob[:,1]
    wwir = agent_ob[:,2]
    wgir = agent_ob[:,3]#
    # print('in env.py, checkxx', wopr.shape)
    # print('in env utils,llllllllllllllllllll',fopt,fwpt,fwit)
    if price is not None:
        oil = price['oil']
        water = price['water']
        inj_w = price['inj_water']
        inj_g = price['inj_gas']
    else:
        print('there is no medium price')

    # print('check output indices in env utils','fopt:',fopt,'fwpt',fwpt,'fwit', fwit, 'fgit','npv',fgit ,fopt * oil + fwpt * water + fwit * inj_w + fgit * inj_g)
    return (wopr * step_time * oil)/10000 

    #return (wopr * step_time * oil + wwpr * step_time * water + wwir * step_time * inj_w + wgir * step_time * inj_g)/10000 * (1/(1+0.06)**((step_num*step_time)/360)) #[n,]

def agg_adj(position_path, data, thrd=0.105 ,self_percent = 0.5):#path, [n,d], number of nerighbor, 筛选邻井的阈值, 自身占比
    p = position_path
    a = pd.read_excel(p).values#[n,2]
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
    final_adj += np.diag([self_percent for i in range(final_adj.shape[0])])
    #整体思路就是让周围井也发挥一半作用，但是邻井被分担的越多，发挥作用越小
    # ind = np.argsort(adj_norm)[:,:n]
    
    indice = data#[n,d]
    
    r_indice = np.matmul(final_adj, indice)#n,d
    # print('in env util agg_adj', indice, final_adj,r_indice )
    
    return r_indice #b

def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)




if __name__ == '__main__':
    path = 'G:/optimiztion_HRL/self_case/CASE.RSM'
    indice = check_indice(path,time_len = 1, key_words = ['WLPR','WWIR','FOPR'], restart = False)
    print(indice)
    # p_act, p_att = 'G:/optimiztion_HRL/SAC_egg/egg/ACTIVE.INC', 'G:/optimiztion_HRL/SAC_egg/egg/EGG_RESTART1.F0002'
    # a,b,c = check_field(p_act, p_att, dim=[7,60,60], squeeze = True)
    # print(a.shape)