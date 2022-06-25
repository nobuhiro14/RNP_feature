import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.dataset import Subset

import sys
import time
import csv 


import sys
from models.runtimeConv2d import DynaConv2d
sys.path.append('../')
from utils.evaluation import *
from utils.early_stop import * 
from test_func import * 


def split_dataset(data_set, split_ratio, order=None):
    n_examples = len(data_set)
    split_at = int(n_examples * split_ratio)
    if split_at < 0:
        raise ValueError('split_at must be non-negative')
    if split_at > n_examples:
        raise ValueError('split_at exceeds the dataset size')

    if order is not None:
        subset1_indices = order[0:split_at]
        subset2_indices = order[split_at:n_examples]
    else:
        subset1_indices = list(range(0,split_at))
        subset2_indices = list(range(split_at,n_examples))

    subset1 = Subset(data_set, subset1_indices)
    subset2 = Subset(data_set, subset2_indices)

    return subset1, subset2


def train_supervised(model,
                    num_epoch,
                    model_num,
                    lr=0.01,
                    batch_size = 64,
                    momentum = 0.9,
                    patience = 3,
                    lasso_flag = 1,
                    num_act = 4, 
                    lasso_param = 1e-5,
                    split_ratio = 0.8,
                    optimizer ="SGD",
                    path = "results",
                    device = "cuda",
                    inits = False):

    #model.train()

    if optimizer == "SGD":
        optimizer = torch.optim.SGD(params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum)
    else :
        sys.stderr.write('Error : Not defined optimizer was chosen')
        return 0

    criterion = nn.CrossEntropyLoss()
    #### schedular parameters ########
    decay_epoch = [120,160]
    decay_factor = 0.1
    schedular = torch.optim.lr_scheduler.MultiStepLR(optimizer,decay_epoch,decay_factor)

    earlystopping = early_stop(patience = patience,path=path)



    ####set up for datasets
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    transform = transforms.Compose([
        #transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
           (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        )
    ])
    dataset = datasets.CIFAR10(
        root= './data', train = True,
        download =True, transform = transform)

    train_dataset, val_dataset  = split_dataset(dataset,split_ratio)

    train_loader = torch.utils.data.DataLoader(train_dataset
        , batch_size = batch_size
        , shuffle = True)
    
    val_loader = torch.utils.data.DataLoader(val_dataset
        , batch_size = batch_size
        , shuffle = True)

    test_dataset = datasets.CIFAR10(
        root= './data', train = False,
        download =True, transform = transform)

    test_loader = torch.utils.data.DataLoader(test_dataset
        , batch_size = batch_size
        , shuffle = True)

    n_total_step = len(train_loader)
    
    #model.train()

    fname = f"{path}/validation_acc_log_No{model_num}.csv"
    f = open(fname, mode="w")
    writer = csv.writer(f)
    
    lasso_cl = GroupLasso(model,lasso_param)
    
    average_time = 0
    mean_loss = 0 
    ct = 0 
    train_log = []
    val_log = []
    tp5_log = []
    ## training phase ##
    for epoch in range(num_epoch):
        running_loss = 0
        start_time = time.time() 
        ls_min = float('inf')
        ls_max = 0
        acc_train  = 0
        acc_tp5 = 0
        acc_val = 0
        for i , (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)

            labels_hat = model(imgs)
            #labels_hat = model(imgs)
            n_corrects = (labels_hat.argmax(axis=1)==labels).sum().item()
            acc_train += 100*(n_corrects/labels.size(0))
            loss_value = criterion(labels_hat,labels)
            _,tp_5_pred = labels_hat.topk(5,1,True,True)
            tp_5_pred = tp_5_pred.t()
            tp5_correct = tp_5_pred.eq(labels.view(1,-1).expand_as(tp_5_pred))
            acc_tp5 += (tp5_correct.float().sum().sum())*(100/batch_size)
            ls_min = min(ls_min,loss_value)
            ls_max = max(ls_max,loss_value)
            if lasso_flag == 1 :
                lasso = lasso_cl.get_group_lasso(model)
            elif lasso_flag == 2: 
                lasso = lasso_cl.get_l1_norm(model,lasso_param)
            elif lasso_flag == 3:
                lasso = lasso_cl.get_group_lasso(model)+ lasso_cl.get_l1_norm(model,lasso_param)
            elif lasso_flag == 4:
                lasso = lasso_cl.get_filter_channel(model)
            elif lasso_flag == 5:
                lasso = lasso_cl.get_linear_lasso(model)
            else :
                lasso = 0.0
            loss_value += lasso 
            loss_value.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss_value.item()
            if (i+1) % 250 == 0:
                print(f"lr : {lr},   top-5 acc : {(tp5_correct.float().sum().sum())*(100/batch_size):.2f}")
                print(f"epoch {epoch+1}/{num_epoch}, step: {i+1}/{n_total_step}: loss = {loss_value:.5f},acc = {100*(n_corrects/labels.size(0)):.2f}%")
                print()
            #schedular.step()
        
        if epoch > num_epoch -10 :
            print("calculate mean loss")
            mean_loss += n_total_step/running_loss 
            ct += 1 
        if epoch %2 == 1 and inits:
            os.makedirs("L2_norm",exist_ok=True)
            measure_L2_norm(model,path,epoch)

        
        end_time = time.time() 
        ver_time = end_time - start_time 
        print(f"times per epoch : {ver_time:.4f} (sec)")
        average_time += ver_time 
        


        ## evaluation phase ##
        _,acc = evaluation(model,val_loader,device)
        writer.writerow([acc_train/n_total_step, acc])

        train_log.append((acc_train/n_total_step))
        tp5_log.append((acc_tp5/n_total_step).to("cpu").detach().numpy())
        val_log.append((acc))



        earlystopping((running_loss/ i),model,ls_min,ls_max)
        if earlystopping.early_stop:
            print("#############Early Stopping################") 
            break 
    
    
    average_time = average_time / (epoch +1)
    print("**************************************")
    print(f"Average training time : {average_time:.4f} (sec)")
    print("**************************************")
    ## evaluation phase ##
    _,acc = evaluation(model,test_loader,device)
    model = earlystopping.model 
    model.Agent = earlystopping.Agent 
    f.close()
    loginfo = {}
    loginfo["train"] = train_log 
    loginfo["tp5"] = tp5_log 
    loginfo["val"] = val_log 

    fname = f"{path}/training_log.mat"
    scipy.io.savemat(fname,loginfo)


    if ct == 0 :
        print(f"ct is zero")
        ct = 10
    mean_loss = mean_loss/ct
    return model, mean_loss




class GroupLasso():
    def __init__(self,model,lb):
        self.lasso = []
        self.linear = []
        self.lb = lb 
        for m in model.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,DynaConv2d):
                tmp = torch.zeros((m.weight.shape[0]),device="cuda")
                tmp_linear = torch.zeros((m.weight.shape[0]),device="cuda")
                weight = m.weight
                for j in range(weight.shape[0]):
                    wi = weight[j]
                    tmp[j] = (lb*torch.tensor(weight[j].numel(),dtype=torch.float).sqrt())
                    tmp_linear[j] = lb *(j+1)
                self.lasso.append(tmp)
                self.linear.append(tmp_linear)
            
    def get_group_lasso(self,model):
        lasso = 0
        i = 0
        for m in model.modules():
                if isinstance(m, nn.Conv2d)or isinstance(m,DynaConv2d):
                    nr = torch.linalg.norm(m.weight,dim=(1,2,3))
                
                    lasso += torch.mul(self.lasso[i],nr).sum()
                    i += 1
                elif isinstance(m,nn.Linear):
                    tmp = m.weight.view(-1) 
                    lmb = self.lb * torch.tensor(tmp.numel(),dtype=torch.float).sqrt()
                    lasso += lmb* torch.sqrt(tmp.square().sum())
        
        return lasso
    
    def get_linear_lasso(self,model):
        lasso = 0
        i = 0
        for m in model.modules():
                if isinstance(m, nn.Conv2d)or isinstance(m,DynaConv2d):
                    nr = torch.linalg.norm(m.weight,dim=(1,2,3))
                
                    lasso += torch.mul(self.linear[i],nr).sum()
                    i += 1
                elif isinstance(m,nn.Linear):
                    tmp = m.weight.view(-1) 
                    lmb = self.lb * torch.tensor(tmp.numel(),dtype=torch.float).sqrt()
                    lasso += lmb* torch.sqrt(tmp.square().sum())
        
        return lasso
        
    def get_l1_norm(self,model,lb):
        return lb*sum(p.view(-1).abs().sum() for p in model.parameters())
    
    def get_filter_channel(self,model):
        lasso = 0
        ch = 0
        
        for m in model.modules():
                if isinstance(m, nn.Conv2d)or isinstance(m,DynaConv2d):
                    nr = torch.linalg.norm(m.weight,dim=(1,2,3))
                    tmp = torch.linalg.norm(m.weight,dim=(0,2,3))
                    ch += torch.mul(self.lb,nr).sum()
                    lasso += torch.mul(self.lb,nr).sum()
                elif isinstance(m,nn.Linear):
                    tmp = m.weight.view(-1) 
                    lmb = self.lb * torch.tensor(tmp.numel(),dtype=torch.float).sqrt()
                    lasso += lmb* torch.sqrt(tmp.square().sum())
        
        return lasso + ch