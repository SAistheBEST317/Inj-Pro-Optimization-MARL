import torch
from torch import nn
import torch.nn.functional as F

class Conv(nn.Module):
    """Multi layer perceptron."""
    def __init__(self, in_dim, out_dim):
        """
        Args: 
            activation_fns: Either list of or a single activation function from torch.nn. If a list is provided, it has to have the same length as units_per_layer. An entry None corresponds to no activation function.
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        conv_list = []
        layer1 = nn.Conv3d(self.in_dim,self.out_dim//16,(4,5,5),(1,1,1),(1,1,1))
        layer2 = nn.Conv3d(self.out_dim//16,self.out_dim//8,(4,5,5),(1,1,1),(1,1,1))
        layer3 = nn.Conv3d(self.out_dim//8,self.out_dim//4,(3,4,4),(1,1,1),(1,1,1))
        layer4 = nn.Conv3d(self.out_dim//4,self.out_dim//2,(3,3,3),(1,1,1),(0,0,0))
        layer5 = nn.Conv3d(self.out_dim//2,self.out_dim,(3,3,3),(1,1,1),(0,0,0))

        conv_list.append(layer1)
        #conv_list.append(layer1_bn)   
        conv_list.append(nn.ReLU())
        conv_list.append(layer2)
        #conv_list.append(layer2_bn)   
        conv_list.append(nn.ReLU())
        conv_list.append(layer3)
        #conv_list.append(layer3_bn)
        conv_list.append(nn.ReLU())

        conv_list.append(layer4)
        #conv_list.append(layer4_bn)  
        conv_list.append(nn.ReLU())
        conv_list.append(layer5)
        conv_list.append(nn.ReLU())
     
        self.model_conv = nn.Sequential(*conv_list)
        self.init_params()

    def init_params(self):
        for layer in self.model_conv:
            if isinstance(layer, nn.Conv3d):
                torch.nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0.01)
                
    def forward(self, x): #goal 只有sat
        # assert x.shape[0] ==200,f'{x.shape}'
        batch_size = x.shape[0]
        out = self.model_conv(x).reshape(batch_size,-1)
        return out
    
class Convtrans(nn.Module):
    """Multi layer perceptron."""
    def __init__(self, in_dim, out_dim):
        """
        Args: 
            activation_fns: Either list of or a single activation function from torch.nn. If a list is provided, it has to have the same length as units_per_layer. An entry None corresponds to no activation function.
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        conv_list = []
        layer1 = nn.ConvTranspose3d(self.in_dim,self.in_dim//2,(6,4,5),(1,1,1),(3,2,2))
        layer2 = nn.ConvTranspose3d(self.in_dim//2,self.in_dim//4,(5,4,4),(1,1,1),(3,2,2))
        layer3 = nn.ConvTranspose3d(self.in_dim//4,self.in_dim//8,(5,4,3),(2,1,1),(2,1,1))
        layer4 = nn.ConvTranspose3d(self.in_dim//8,self.out_dim,(4,3,3),(1,0,0),(2,2,2))

        conv_list.append(layer1)
        conv_list.append(nn.ELU())
        conv_list.append(layer2)
        conv_list.append(nn.ELU())
        conv_list.append(layer3)
        conv_list.append(nn.ELU())
        conv_list.append(layer4)
        conv_list.append(nn.tanh())#最后一层一定要保证>0 因为 后面要嵌套torch.distribution.norm() 内部的scale参数必须大于0
        self.model_convtrans = nn.Sequential(*conv_list)
        self.init_params()
        
    def init_params(self):
        for layer in self.model_convtrans:
            if isinstance(layer, nn.ConvTranspose3d):
                torch.nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0.01)

    def forward(self, x): #goal 只有sat
        x_field = x.view(x.shape[0], x.shape[1],1,1,1)
        out = self.model_convtrans(x_field)
        
        return out
    
class Mlp_out(nn.Module):
    """Multi layer perceptron."""
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers):
        """
        Args: 
            activation_fns: Either list of or a single activation function from torch.nn. If a list is provided, it has to have the same length as units_per_layer. An entry None corresponds to no activation function.
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        
        #输出结果
        self.layers = []
        for n in range(n_layers-1):
            self.layers.append(nn.Linear(self.in_dim, self.hidden_dim))
            self.layers.append(nn.Tanh())
            self.in_dim = self.hidden_dim
        self.layers.append(nn.Linear(self.hidden_dim, self.out_dim)) 
        # self.layers.append(nn.Tanh()) 
        self.mlp = nn.Sequential(*self.layers)

        self.init_params()
        
    def init_params(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0.01)
    def forward(self, x): #goal 只有sat
        out = self.mlp(x)
        return out

class Mlp_in(nn.Module):
    """Multi layer perceptron."""
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers):
        """
        Args: 
            activation_fns: Either list of or a single activation function from torch.nn. If a list is provided, it has to have the same length as units_per_layer. An entry None corresponds to no activation function.
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        
        #输出结果
        self.layers = []
        for n in range(n_layers-1):
            self.layers.append(nn.Linear(self.in_dim, self.hidden_dim))
            self.layers.append(nn.ELU())
            self.in_dim = self.hidden_dim
        self.layers.append(nn.Linear(self.hidden_dim, self.out_dim)) 
        self.layers.append(nn.Tanh()) 
        self.mlp = nn.Sequential(*self.layers)

        self.init_params()
        
    def init_params(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0.01)
                
    def forward(self, x): #goal 只有sat
        out = self.mlp(x)
        return out
class W_Conv(nn.Module):#用于单井观测的卷积
    """Multi layer perceptron."""
    def __init__(self, in_dim, out_dim):
        """
        Args: 
            activation_fns: Either list of or a single activation function from torch.nn. If a list is provided, it has to have the same length as units_per_layer. An entry None corresponds to no activation function.
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        conv_list = []
        layer1 = nn.Conv3d(self.in_dim,2*self.in_dim,(4,4,4),(1,2,2),1) #in 1, out1, kernel(C,W,H) 2,3,3, stride 2, padding 1
        layer2 = nn.Conv3d(2*self.in_dim,4*self.in_dim,(4,4,4),(1,2,2),(0,1,1))# z 方向收敛到1
        layer3 = nn.Conv3d(4*self.in_dim,8*self.in_dim,(3,3,3),1,(0,1,1))
        layer4= nn.Conv3d(8*self.in_dim, 16*self.in_dim,(1,3,3),1,0)
        layer5= nn.Conv3d(16*self.in_dim, self.out_dim,(1,3,3),1,0)
        conv_list.append(layer1)
        conv_list.append(nn.ELU())
        conv_list.append(layer2)
        conv_list.append(nn.ELU())
        conv_list.append(layer3)
        conv_list.append(nn.ELU())
        conv_list.append(layer4)
        conv_list.append(nn.ELU())
        conv_list.append(layer5)
        conv_list.append(nn.Tanh())
        self.model_conv = nn.Sequential(*conv_list)
        self.init_params()

    def init_params(self):
        for layer in self.model_conv:
            if isinstance(layer, nn.Conv3d):
                torch.nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0.01)
                
    def forward(self, x): #goal 只有sat
    
        out = self.model_conv(x)
        return out