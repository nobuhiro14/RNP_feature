import numpy as np

class Reward():
    def __init__(self,p,len_action):
        self.p = p
        self.alpha = None
        self.rewards = np.zeros((len_action,))
    
    def initialize(self,alpha):
        self.alpha = alpha 
        

    def get_reward(self,actions,done,loss):
        
        self.rewards =-1*( done*self.alpha*loss + self.alpha*actions )

        return self.rewards
