import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import sys 

sys.path.append('../')
from models.base_model import base_model
from models.Agent import Agent
from models.runtimeConv2d import DynaConv2d


def redistribution(model,device="cuda"):
    model.redistribute_act()
    new_model = base_model(model.in_channels,model.num_act,model.ch_param.data).to(device)
    new_model.Agent = model.Agent 
    parameters = []
    linears  = []
    ch = model.channels
    print(ch)
    i = 0
    for m in model.modules():
        if isinstance(m,nn.Conv2d) :
            parameters.append(m.weight)
        elif isinstance(m,DynaConv2d):
            parameters.append(m.weight[0:ch[i]][0:ch[i+1]])
            i+= 1 
        elif isinstance(m, nn.Linear):
            linears.append(m.weight)
    
    i = 0
    j = 0
    for m in new_model.modules():
        if isinstance(m,nn.Conv2d) or isinstance(m,DynaConv2d):
            m.weight = nn.Parameter(parameters[i])
            i+= 1 
        
        elif isinstance(m, nn.Linear):
            m.weight = nn.Parameter(linears[j])
            j+= 1 
    
    return new_model
    
def load_model(pth,in_channels,num_act,device = "cuda"):
    m_dict = torch.load(pth)
    model = base_model(in_channels,num_act,m_dict["ch_param"]).to(device)
    model.load_state_dict(m_dict)

    return model

