import numpy as np
import os
import time
import scipy.io
import json
import sys
from collections import namedtuple
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


from models.base_model import base_model
from models.benchmark_models import bench_model
from models.base_model_cnn import base_model_cnn
from models.Agent import Agent
from utils.options import *
from test_func import *


LEARNING_RATE = 1e-6
EPS = 0.1
ACTIONS_PER_EPISODE = 12
ALPHA = 0.95
def main():
    opt = TestOptions()
    args = opt.get_args()
    reader = open(f"{args.load_pth}/params.json","r")
    params = json.load(reader)

    #summary(model,(3,224,224))

    ############# dataset preparation #############
    transform = transforms.Compose([
        #transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
           (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        )
    ])
    test_dataset = datasets.CIFAR10(
        root= './data', train = False,
        download =True, transform = transform)

    test_loader = torch.utils.data.DataLoader(test_dataset
        , batch_size = 1
        , shuffle = True)

    ############# model initialization #############
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if args.load_vgg16 :
        model = models.vgg16(pretrained=False).to(device)
        model_pth = f"{args.load_pth}/vgg16.model"
        model.load_state_dict(torch.load(model_pth))
        model_profiler(model,test_loader,device,args.load_pth)
    else :
        if args.use_profiler :
            model = bench_model(params["in_channels"],params["num_actions"],device=device)
        else :
            model = base_model(params["in_channels"],params["num_actions"],device=device)
            
        states = max(model.channels)
        agent = Agent(params["in_channels"],params["num_actions"],states)

        with open(f"{args.load_pth}/accuracy.csv","w+") as f :
            writer = csv.writer(f)
            ############# evaluation #############
            for num in range(1,params["train_num"]+1):
                model_pth = f"{args.load_pth}/base_model_No{num}.model"
                model.load_state_dict(torch.load(model_pth))
                #model.channels = model.ch_param.data.int().tolist()
                Agent_pth = f"{args.load_pth}/Agent_model_No{num}.model"
                agent.load_state_dict(torch.load(Agent_pth))
                model.Agent = agent.to(device)
                model.Agent.to(device)
                model = model.to(device)
                model_summary(model,bt_size=64)
                model.set_Agent_mode("supervised")
                if args.use_profiler :
                    model_profiler(model,test_loader,device,args.load_pth)
                else :
                    acc = action_distribution(model, test_loader,num,params["num_actions"],ACTIONS_PER_EPISODE,args.load_pth, device="cuda")
                    measure_L2_norm(model,args.load_pth,num)
                    #print(model.ch_param)
                    print(model.channels)
                    writer.writerow([f"model_num : {num}", acc])








if __name__ == '__main__':
    main()