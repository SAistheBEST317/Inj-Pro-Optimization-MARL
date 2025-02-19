#  A Multi-Agent Reinforcement Learning Framework for Scalable and Adaptive InjectionProduction Optimization in Reservoir Management

The code for paper "A Multi-Agent Reinforcement Learning Framework for Scalable and Adaptive Injection-Production Optimization in Reservoir Management". 

### Authors
* First Author / Corresponding Author: Shuang Zhao, Master, The University of Hong Kong, HONG KONG, u3640969@connect.hku.hk
* Second Author: Chengcheng Liu, School of Civil Engineering, Qingdao University of Technology, Shandong, China, xyyhwhc@163.com
* Third Author: Ji Ma, Communist Party of China Shengli Petroleum Management Bureau Co., Ltd., Shandong, China, maji192.slyt@sinopec.com

### Abstract
The growing global energy demand necessitates more efficient reservoir management, highlighting the need for advanced injection-production optimization strategies. Traditional methods often suffer from high computational costs and poor scalability, particularly when new wells are introduced, as they require retraining of nearby agents. These limitations hinder their application in large, heterogeneous reservoirs and reduce their effectiveness for adaptive decision-making. To address these challenges, this study introduces an enhanced multi-agent reinforcement learning (MARL) framework with three key innovations. First, an online-updating surrogate model, based on a simplified U-Net architecture, partially replaces costly numerical simulations, significantly reducing computational overhead. Second, a regional observation construction method encodes relative well positions to capture inter-well dependencies and enhance local decision-making. Third, a self-adaptive, graph-based unified agent design eliminates the need for retraining when new wells are added, ensuring scalability. The proposed framework was validated using both a synthetic “Three-Channel” model and a real oilfield case. Experimental results show substantial improvements in displacement efficiency, delayed water breakthrough, and increased Net Present Value (NPV). Additionally, the framework adapts seamlessly to the addition of new wells without retraining, maintaining high computational efficiency. These results underscore the practical potential of the MARL-based approach as a robust and flexible solution for real-time reservoir management in dynamically evolving oil fields.

### Installation 

1- Get the repository
```bash
git clone https://github.com/SAistheBEST317/MARL.git
```
2- Install the dependencies 

```bash
python -m venv myvenv
source myvenv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Usage

#### Model Option

| name | type | description | default |
|:-------|:-------:|-------:|-------:|
| --max_steps    | int   | number of steps in one episode                                                          |  default=12    |
| --interval     | int   | number of days in every timestep                                                        |  default=60    |
| --init_step    | int   | initial time step                                                                       |  default=1     |
| --well_num     | int   | Number of new Wells + number of old Wells + number of features of a new well location   |  default=4     |
| --n_del_well   | int   | Number of new Wells + number of old Wells + number of features of a new well location   |  default= 2    |
| --n_new_well   | int   | Number of new Wells + number of old Wells + number of features of a new well location   |  default= 2    |
| --n_old_well   | int   | Number of new Wells + number of old Wells + number of features of a new well location   |  default=12    |
| --well_range   | int   | number of adjacent Wells that received interference                                     |  default= 5    |
| --reward_dist  | int   | The upper range of the adjacent well for which the reward is calculated                 |  default=20    |
| --state_dim    | int   | well_range * dim_num                                                                    |  default=(20,) |
| --state_c_dim  | int   | Number of field data channels (number of fields)                                        |  default= 4    |

| --well_style   | str   | Type of well                                                                            |default=vertical                              |
| --path         | str   | number of days in every timestep                                                        |default='eclipse_model/EGG/EGG'               |
| --posi_path    | str   | initial time step                                                                       |default='eclipse_model/EGG/well_position.xlsx'|
| --new_well_loc | str   | Old well coordinates                                                                    |default='eclipse_model/EGG/well_position.xlsx'|
| --new_path     | str   | eclipse model path                                                                      |default= 'eclipse_model/EGG/EGG'              |
| --action_record| str   | Record the action and initial status                                                    |  default= 'save/action_record'               |
| --n_old_well   | int   | Number of new Wells + number of old Wells + number of features of a new well location   |  default=12                                  |
| --well_list    | int   | number of adjacent Wells that received interference                                     |  default= 5                                  |
| --reward_dist  | int   | The upper range of the adjacent well for which the reward is calculated                 |  default=['INJECT1','INJECT2','INJECT3','INJECT4','INJECT5','INJECT6','INJECT7','INJECT8','PROD1','PROD2','PROD3','PROD4']   |

#### 
