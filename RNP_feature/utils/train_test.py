import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
import time
import csv 


import sys
sys.path.append('../')
from utils.evaluation import *
from utils.early_stop import * 



def train_supervised(model,
                    num_epoch,
                    model_num,
                    lr=0.01,
                    batch_size = 64,
                    momentum = 0.9,
                    patience = 3,
                    add_gl_norm = True,
                    num_act = 4, 
                    lasso_param = [1,1,1,1],
                    optimizer ="SGD",
                    path = "results",
                    device = "cuda"):

    model.train()

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
    train_dataset = datasets.CIFAR10(
        root= './data', train = True,
        download =True, transform = transform)

    train_loader = torch.utils.data.DataLoader(train_dataset
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
    
    average_time = 0
    ## training phase ##
    for epoch in range(num_epoch):
        running_loss = 0
        start_time = time.time() 
        ls_min = float('inf')
        ls_max = 0
        acc_train  = 0
        for i , (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)

            labels_hat = model(imgs)
            #labels_hat = model(imgs)
            n_corrects = (labels_hat.argmax(axis=1)==labels).sum().item()
            acc_train += 100*(n_corrects/labels.size(0))
            loss_value = criterion(labels_hat,labels)
            ls_min = min(ls_min,loss_value)
            ls_max = max(ls_max,loss_value)
            if add_gl_norm :
                #lasso = group_norm(model,lasso_param,num_act)
                lasso = norm(model)
            else :
                lasso = torch.tensor(0.0)
            loss_value += lasso 
            loss_value.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss_value.item()
            if (i+1) % 250 == 0:
                print(f"epoch {epoch+1}/{num_epoch}, step: {i+1}/{n_total_step}: loss = {loss_value:.5f},acc = {100*(n_corrects/labels.size(0)):.2f}%")
                print()
            #schedular.step()

        
        end_time = time.time() 
        ver_time = end_time - start_time 
        print(f"times per epoch : {ver_time:.4f} (sec)")
        average_time += ver_time 



        ## evaluation phase ##
        _,_,acc = evaluation(model,test_loader,device)
        writer.writerow([acc_train/n_total_step, acc])

    
    average_time = average_time / (epoch +1)
    print("**************************************")
    print(f"Average training time : {average_time:.4f} (sec)")
    print("**************************************")

    
    f.close()





    return model,ls_min, ls_max 



def group_norm(model,gl_lambda,num_blk):
    gl_reg = torch.tensor(0., dtype=torch.float32).cuda()
    for param in model.parameters():
        dim = param.size()
        if dim.__len__() == 4 :
            div1 = list(torch.chunk(param,int(num_blk),0))
            all_blks = []
            for i, div2 in enumerate(div1):
                temp = list(torch.chunk(div2,int(num_blk),1))
                for blk in temp:
                    all_blks.append(blk)
            for l2_param in all_blks:
                gl_reg += float(gl_lambda[i])* torch.norm(l2_param, 2)
    return gl_reg 


def norm(model,gl_lambda = 1):
    gl_reg = torch.tensor(0., dtype=torch.float32).cuda()
    for param in model.parameters():
        dim = param.size() 
        if dim.__len__() == 4:
            gl_reg += torch.norm(param,1)
    
    return gl_reg *float(gl_lambda)
