# from .develop_env import development#前面的点表示当前文件夹，否则就会从主文件文件夹当作路径
from gym.envs.registration import register
from .develop_q_v1 import development_q1
from .develop_q_v2 import development_q2
from .develop_q_v3 import development_q3
from .develop_q_v4 import development_q4

from .develop_p_v1 import development_p1

from .develop_qt_v1 import development_qt1
from .develop_qt_v2 import development_qt2
from .develop_qt_v3 import development_qt3
from .develop_qt_v4 import development_qt4

from .develop_qtp_v1 import development_qtp1
from .develop_qtp_v2 import development_qtp2
from .develop_qtp_v3 import development_qtp3
from .develop_qtp_v4 import development_qtp4

from .develop_qtpth_v1 import development_qtpth1
from .develop_qtptv_v1 import development_qtptv1 
from .develop_qtptv_v2 import development_qtptv2 
from .develop_qtptv_v3 import development_qtptv3 
from .develop_qtptv_v4 import development_qtptv4 

from .develop_qt_hybrid_v1 import development_qt_hybrid_v1
from .develop_her_v4 import development_her_v4
from .develop_qt_marl_v4 import development_qt_marl_v4

# from .session.session import Session

# register test envs with gym
print('register develop environment')
# register(
#         id = "develop-v0",
#         entry_point = "envs:development" #注册时必须要加上：，前面是从主文件开始导入包的路径，后面的env名字
#         )
register(
        id = "develop_q_v1.0",
        entry_point = "environments:development_q1" #注册时必须要加上：，前面是从主文件夹名字开始导入包的路径，后面的env名字
        )
register(
        id = "develop_q_v2.0",
        entry_point = "environments:development_q2" #注册时必须要加上：，前面是从主文件开始导入包的路径，后面的env名字
        )
register(
        id = "develop_q_v3.0",
        entry_point = "environments:development_q3" #注册时必须要加上：，前面是从主文件开始导入包的路径，后面的env名字
        )
register(
        id = "develop_q_v4.0",
        entry_point = "environments:development_q4" #注册时必须要加上：，前面是从主文件开始导入包的路径，后面的env名字
        )
register(
        id = "develop_p_v1.0",
        entry_point = "environments:development_p1" #注册时必须要加上：，前面是从主文件开始导入包的路径，后面的env名字
        )
register(
        id = "develop_qt_v1.0",
        entry_point = "environments:development_qt1" #注册时必须要加上：，前面是从主文件开始导入包的路径，后面的env名字
        )
register(
        id = "develop_qt_v2.0",
        entry_point = "environments:development_qt2" #注册时必须要加上：，前面是从主文件开始导入包的路径，后面的env名字
        )
register(
        id = "develop_qt_v3.0",
        entry_point = "environments:development_qt3" #注册时必须要加上：，前面是从主文件开始导入包的路径，后面的env名字
        )
register(
        id = "develop_qt_v4.0",
        entry_point = "environments:development_qt4" #注册时必须要加上：，前面是从主文件开始导入包的路径，后面的env名字
        )

register(
        id = "develop_qtp_v1.0",
        entry_point = "environments:development_qtp1" #注册时必须要加上：，前面是从主文件开始导入包的路径，后面的env名字
        )

register(
        id = "develop_qtp_v2.0",
        entry_point = "environments:development_qtp2" #注册时必须要加上：，前面是从主文件开始导入包的路径，后面的env名字
        )

register(
        id = "develop_qtp_v3.0",
        entry_point = "environments:development_qtp3" #注册时必须要加上：，前面是从主文件开始导入包的路径，后面的env名字
        )

register(
        id = "develop_qtp_v4.0",
        entry_point = "environments:development_qtp4" #注册时必须要加上：，前面是从主文件开始导入包的路径，后面的env名字
        )

register(
        id = "develop_qtpth_v1.0",
        entry_point = "environments:development_qtpth1" #注册时必须要加上：，前面是从主文件开始导入包的路径，后面的env名字
        )
register(
        id = "develop_qtptv_v2.0",
        entry_point = "environments:development_qtptv2" #注册时必须要加上：，前面是从主文件开始导入包的路径，后面的env名字
        )
register(
        id = "develop_qtptv_v3.0",
        entry_point = "environments:development_qtptv3" #注册时必须要加上：，前面是从主文件开始导入包的路径，后面的env名字
        )
register(
        id = "develop_qtptv_v4.0",
        entry_point = "environments:development_qtptv4" #注册时必须要加上：，前面是从主文件开始导入包的路径，后面的env名字
        )
register(
        id = "develop_qtptv_v1.0",
        entry_point = "environments:development_qtptv1" #注册时必须要加上：，前面是从主文件开始导入包的路径，后面的env名字
        )

register(
        id = "develop_qt_hybrid_v1.0",
        entry_point = "environments:development_qt_hybrid_v1" #注册时必须要加上：，前面是从主文件开始导入包的路径，后面的env名字
        )

register(
        id = "develop_her_v4.0",
        entry_point = "environments:development_her_v4" #注册时必须要加上：，前面是从主文件开始导入包的路径，后面的env名字
        )
register(
        id = "develop_qt_marl_v4.0",
        entry_point = "environments:development_qt_marl_v4" #注册时必须要加上：，前面是从主文件开始导入包的路径，后面的env名字
        )


print('import envs sucess')

# import gym

# a = gym.make("develop_qtptv_v1.0")
# print(a)