import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision import models
from torch import nn,optim
import numpy as np
import copy 

class early_stop : 

    def __init__(self,patience=5, verbose=True, path = "results"):

        self.patience = patience 
        self.verbose = verbose 
        self.counter = 0 
        self.best_score = None 
        self.early_stop = False 
        self.val_loss_min = np.Inf 
        self.path = f"{path}/checkpoint_model.pth" 
        self.model = None
        self.Agent = None 
        self.ls_min = None 
        self.ls_max = None 


    
    def __call__(self,val_loss,model,ls_min,ls_max):

        score = -val_loss 

        if self.best_score is None :
            self.best_score = score 
            self.checkpoint(val_loss,model)
            self.ls_min = ls_min
            self.ls_max = ls_max 
        elif score < self.best_score: 
            self.counter += 1 
            if self.verbose : 
                print(f"EarlyStopping counter : {self.counter} out of {self.patience}")
            if self.counter >= self.patience : 
                self.early_stop = True 
        
        else :
            self.best_score = score 
            self.checkpoint(val_loss,model) 
            self.counter = 0
            self.ls_min = ls_min
            self.ls_max = ls_max 
        
    
    def checkpoint(self,val_loss, model):

        if self.verbose :
            print(f"Validation loss decreased( { self.val_loss_min: .6f} --> {val_loss: .6f}. Saving model ...")
        torch.save(model.state_dict(),self.path)
        self.val_loss_min = val_loss 
        self.model = copy.deepcopy(model )
        self.Agent = copy.deepcopy(model.Agent)
            