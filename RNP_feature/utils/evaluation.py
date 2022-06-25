import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys


def evaluation(model, test_loader,device = "cuda"):




    criterion = nn.CrossEntropyLoss()


    model = model.to(device)
    n_total_step = len(test_loader)
    model.eval()

    ## evaluation phase ##
    with torch.no_grad():
        number_corrects = 0
        number_samples = 0
        mean_loss =0
        acc_tp5 = 0
        
        for i, (test_images_set , test_labels_set) in enumerate(test_loader):
            test_images_set = test_images_set.to(device)
            test_labels_set = test_labels_set.to(device)
            batch_size = test_images_set.shape[0]

            y_predicted = model(test_images_set)
            loss_value = criterion(y_predicted,test_labels_set).item()
            mean_loss += loss_value
            labels_predicted = y_predicted.argmax(axis = 1)
            _,tp_5_pred = y_predicted.topk(5,1,True,True)
            tp_5_pred = tp_5_pred.t()
            tp5_correct = tp_5_pred.eq(test_labels_set.view(1,-1).expand_as(tp_5_pred))
            acc_tp5 += (tp5_correct.float().sum().sum())*(100/batch_size)
            number_corrects += (labels_predicted==test_labels_set).sum().item()
            number_samples += test_labels_set.size(0)
        print(f"Overall accuracy {(number_corrects / number_samples)*100}%")
        print(f"top 5 accuracy : {acc_tp5/n_total_step:2f}%")

    acc  = (number_corrects / number_samples)*100

    mean_loss =  n_total_step/mean_loss
    


    return mean_loss,acc 