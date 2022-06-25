import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms



class DynaConv2d(nn.Module):
    def __init__(self,):
        super(DynaConv2d, self).__init__()
