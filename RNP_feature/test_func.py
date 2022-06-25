import sys
import numpy as np
import os
import time
import scipy.io
from collections import namedtuple


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from models.runtimeConv2d import DynaConv2d

from torchinfo import summary


def measure_L2_norm(model,save_pth,model_num):
    results = {}
    #ls_result = []
    num = 1
    ch = model.channels
    print(ch)
    result_div = {}
    i = 0
    for m in model.modules():
        if isinstance(m,DynaConv2d):
            div_weight = m.weight[0:ch[i]]
            i += 1 
            div_tmp = torch.linalg.norm(div_weight,dim=(1,2,3))
            
            tmp = torch.linalg.norm(m.weight,dim=(1,2,3))
            name = f"conv_No{num:02}"
            results[name] = tmp.cpu().detach().numpy()
            result_div[name] = div_tmp.cpu().detach().numpy()
            #ls_result.append(tmp)
            num +=1
    if model_num == -1 :
        fname = f"{save_pth}/L2_norm_result_base_model.mat"
        scipy.io.savemat(fname,{f"norm_base":results})
    elif model_num == -2:
        fname = f"{save_pth}/L2_norm_result_loaded_vgg16model.mat"
        scipy.io.savemat(fname,{f"norm_base":results})
    else :
        fname = f"{save_pth}/L2_norm_result_No{model_num}.mat"
        div_fname = f"{save_pth}/L2_norm_div_No{model_num}.mat"
        
        scipy.io.savemat(fname,{f"norm{model_num}":results})
        scipy.io.savemat(div_fname,{f"div{model_num}":result_div})
    #scipy.io.savemat(fname,{f"norm{model_num}":ls_result})



def action_distribution(model, datasets,model_num,act_num,ep_num,path, device="cuda"):
    number_corrects = 0
    number_samples = 0
    ave_action = 0
    acc_tp5 = 0 
    n_length  = len(datasets)
    dist = np.zeros((ep_num,act_num))

    for i, (test_images_set , test_labels_set) in enumerate(datasets):
        test_images_set = test_images_set.to(device)
        test_labels_set = test_labels_set.to(device)
        batch_size = test_images_set.shape[0]

        y_predicted,_,actions,_ ,_= model.forward_rl(test_images_set)
        labels_predicted = y_predicted.argmax(axis = 1)
        number_corrects += (labels_predicted==test_labels_set).sum().item()
        _,tp_5_pred = y_predicted.topk(5,1,True,True)
        tp_5_pred = tp_5_pred.t()
        tp5_correct = tp_5_pred.eq(test_labels_set.view(1,-1).expand_as(tp_5_pred))
        acc_tp5 += (tp5_correct.float().sum().sum())*(100/batch_size)
        number_samples += test_labels_set.size(0)
        ave_action += actions
        for ep in range(ep_num):
            dist[ep][int(actions[ep])] +=1

    print(dist)

    os.makedirs(path,exist_ok=True)
    fname = f"{path}/distribution_No{model_num}.mat"
    dic_name = f"dis{model_num}"
    scipy.io.savemat(fname,{dic_name:dist})
    accuracy = (number_corrects / number_samples)*100
    print(f"Overall accuracy {(number_corrects / number_samples)*100}%")
    print(f"Top 5 acc : {acc_tp5/n_length} %")
    ave_action = ave_action/(i+1)
    print(ave_action)
    return accuracy 


def model_profiler(model,dataset,device,dir_name):
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    number_samples = 0
    number_corrects =0
    n_total_step = len(dataset)

    with torch.profiler.profile(
        activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=2,
            warmup=2,
            active=6,
            repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_name),
        with_flops = True
        ) as profiler:

        for i, (test_images_set , test_labels_set) in enumerate(dataset):
            test_images_set = test_images_set.to(device)
            test_labels_set = test_labels_set.to(device)


            y_predicted = model(test_images_set)
            loss_value = criterion(y_predicted,test_labels_set).item()
            labels_predicted = y_predicted.argmax(axis = 1)
            number_corrects += (labels_predicted==test_labels_set).sum().item()
            number_samples += test_labels_set.size(0)
            profiler.step()

            if i % 500 == 0:
                print(f"{(i/n_total_step)*100}")
        print(f"Overall accuracy {(number_corrects / number_samples)*100}%")

    print(profiler.key_averages().table(sort_by="cuda_time", row_limit=-1))

def model_summary(model,bt_size):
    print(f"Batch size for summary : {bt_size}")
    print(summary(
        model,
        input_size=(bt_size,3,64,64),
        col_names = ["output_size", "num_params"],
    ))
