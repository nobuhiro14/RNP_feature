import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import numpy as np
import os
import json
from collections import namedtuple

from models.base_model import base_model
from models.base_model_cnn import base_model_cnn
from models.Agent import Agent
from utils.train_supervised import *
from utils.train_reinforcement import *
from utils.evaluation import *
from utils.options import *
from utils.reward import *
from test_func import *
from utils.functions import * 

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])




ACTIONS_PER_EPISODE =12

def main():
    opt = Options()
    args = opt.get_args()
    os.makedirs(f"{args.save_pth}",exist_ok=True)
    with open(f"{args.save_pth}/params.json",mode="w") as f:
        json.dump(args.__dict__, f, indent=4)
    ############## model initialization ########################
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = base_model(args.in_channels,args.num_actions,device=device)
    Q_target = Agent(args.in_channels,args.num_actions)
    Q_target = Q_target.to(device)
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = True
    


    Rewards = Reward(args.p_ls,ACTIONS_PER_EPISODE)
    optimizer = OptimizerSpec(
        constructor=torch.optim.RMSprop,
        kwargs=dict(lr=args.lr_re, alpha=args.alpha_re, eps=args.eps_re)
    )


    ############# prepare dataset ##############################
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


    ################## initial training starts ###########################
    if args.init_training ==0 :
        print("initial training for base_model starts")
        print("**************************************")
        lr_init = 0.01
        model.set_Agent_mode("init_training")
        model,mean_loss = train_supervised(model,
                                args.epoch,
                                0,
                                lr_init,
                                args.bt_size_sp,
                                args.momentum,
                                args.num_patience,
                                args.lasso_flag,
                                args.num_actions,
                                args.lasso_param,
                                path = args.save_pth,
                                device =device,
                                inits = True)
        model_save_path = f"{args.save_pth}/only_base_model.model"
        torch.save(model.state_dict(), model_save_path)
        print("initial training for base_model ends")
    elif args.init_training == 1 :
        ################## load pretrained model ###########################
        print("load the trained base model")
        print("**************************************")
        train_dataset = datasets.CIFAR10(
            root= './data', train = True,
            download =True, transform = transform)
        train_loader  = torch.utils.data.DataLoader(train_dataset
            , batch_size = args.bt_size_sp
            , shuffle = True)

        model_save_path = f"{args.save_pth}/only_base_model.model"
        model.load_state_dict(torch.load(model_save_path))
        model.Agent = Agent(args.in_channels,args.num_actions)
        model.set_Agent_mode("init_training")
        mean_loss, _= evaluation(model,train_loader,device)
    elif args.init_training == 2 :
        ################## load pretrained model ###########################
        print("load the vgg16 model")
        print("**************************************")
        train_dataset = datasets.CIFAR10(
            root= './data', train = True,
            download =True, transform = transform)
        train_loader  = torch.utils.data.DataLoader(train_dataset
            , batch_size = args.bt_size_sp
            , shuffle = True)
        vgg_pth = f"{args.save_pth}/vgg16.model"
   
        load_vgg_parameter(model,vgg_pth,device)
        for param in model.parameters():
            param.requires_grad = True

        measure_L2_norm(model,args.save_pth,-2)
        
        model.set_Agent_mode("init_training")
        mean_loss,_ = evaluation(model,train_loader,device)

        
        



    ######################## repeat training starts ###############################
    for i in range(args.train_num):
        print("**************************************")
        print(f"starts repeat training No: {i+1}/{args.train_num}")
        print("**************************************")

        Rewards.initialize(mean_loss)
        print(model.channels)

        ################### reinforcement learning #############################
        model.set_Agent_mode("reinforcement")
        model , Q_target = train_reinforcement(model,
                                            Q_target,
                                            args.epsilon_re,
                                            optimizer,
                                            Rewards,
                                            ACTIONS_PER_EPISODE,
                                            i+1,
                                            args.in_channels,
                                            args.num_actions,
                                            args.buffer_size,
                                            args.bt_size_re,
                                            args.lr_freq_re,
                                            args.lr_start_re,
                                            args.update_re,
                                            args.activate_decay,
                                            path=args.save_pth,
                                            device = device)
        model_save_path = f"{args.save_pth}/Agent_after_re_No{i}.model"
        torch.save(model.Agent.state_dict(), model_save_path)

        ################### evaluation phase #############################
        model.set_Agent_mode("supervised")
        action_distribution(model,
                            test_loader,
                            0,
                            args.num_actions,
                            ACTIONS_PER_EPISODE,
                            args.save_pth,
                            device=device)



        ################### supervised learning #############################
        model,mean_loss = train_supervised(model,
                            args.epoch,
                            i+1,
                            args.lr_sp,
                            args.bt_size_sp,
                            args.momentum,
                            args.num_patience,
                            args.lasso_flag,
                            args.num_actions,
                            args.lasso_param,
                            path = args.save_pth,
                            device =device)

        ################### evaluation phase #############################
        action_distribution(model,
                            test_loader,
                            i+1,
                            args.num_actions,
                            ACTIONS_PER_EPISODE,
                            args.save_pth,
                            device=device)
        
        

        model_save_path = f"{args.save_pth}/base_model_No{i+1}.model"
        torch.save(model.state_dict(), model_save_path)
        model_save_path = f"{args.save_pth}/Agent_model_No{i+1}.model"
        torch.save(model.Agent.state_dict(), model_save_path)



def load_vgg_parameter(model,vgg_pth,device):
    vgg16 = models.vgg16().to(device)
    vgg16.load_state_dict(torch.load(vgg_pth))
    parameters = []
    for m in vgg16.modules():
        if isinstance(m, nn.Conv2d):
            parameters.append(m.weight)
    i = 0    
    print(len(parameters))
    for m in model.modules():
        if isinstance(m,nn.Conv2d) or isinstance(m,DynaConv2d):
            
            m.weight = parameters[i]
            i+= 1 
    
    return model 




if __name__ == '__main__':
    main()
