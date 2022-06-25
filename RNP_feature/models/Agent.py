import torch
import torch.nn as nn
import torch.functional as F
from models.time_distributed import *
from torch.nn.utils.rnn import pack_padded_sequence


class Agent(nn.Module):
    def __init__(self,in_channels, num_actions,flag = False,device="cuda"):
        '''
        in_channels : channel size of hidden layer at DQN
        num_actions : choices of actions at each states
        flag : indicates whether on-training or not
        '''
        super(Agent, self).__init__()
        #parameters
        self.ch = in_channels
        self.num_act = num_actions
        self.device = device
        self.is_training = flag
        self.lowest_epsilon = None
        self.num_epochs = None
        self.epsilon = None




        #network structure
        self.timedist = TimeDistributed(nn.Sequential(nn.Linear(1,self.ch,bias = True),nn.ReLU()))
        self.rnn = nn.RNN(input_size=self.ch,hidden_size=self.ch,nonlinearity="relu",batch_first=True)
        self.linear2 = nn.Linear(self.ch,self.num_act,bias=True)
        self.prob = nn.Sigmoid()

        self.weight_initialization()
    
    def weight_initialization(self):
        nn.init.kaiming_normal_(self.timedist.module[0].weight)
        #nn.init.kaiming_normal_(self.rnn.weight)
        nn.init.kaiming_normal_(self.linear2.weight)


    def forward(self,x):
        '''
        input shape : (batch, timesteps, input_size)

        '''

        '''
        x = x.mean(dim=(-2,-1))
        k, ch  = x.shape
        zero_pad = torch.zeros((k,self.num_states-ch),device=self.device)
        x = torch.cat([x,zero_pad],dim=1)
        '''
        x = self.timedist(x)
        _ ,x= self.rnn(x.float())
        y = self.linear2(x[0])
        #y = self.prob(y)

        return y

    def Action(self,q_values):
        if self.is_training  == True :
            ## epsilon greedy methods
            if torch.rand(1) < self.epsilon:  # random行動
                action = torch.randint(0, self.num_act,(q_values.shape[0],) )
            else:   # greedy 行動
                action = torch.argmax(q_values,dim=-1)
        else :
            action = torch.argmax(q_values,dim=-1)

        return action

    def _decay_epsilon(self,epsilon=None,lowest=None,decay_step=None):
        if epsilon ==None and decay_step ==None :
            self.epsilon = max(self.lowest_epsilon, self.epsilon-self.decay_epsilon)

        else :
            self.epsilon = epsilon
            self.lowest_epsilon = lowest
            self.decay_epsilon = 1.0/decay_step 
