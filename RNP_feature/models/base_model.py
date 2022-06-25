import torch
import torch.nn as nn
import torch.functional as F
from .runtimeConv2d import DynaConv2d
from .Agent import Agent


class base_model(nn.Module):
    def __init__(self,in_channels, num_actions,ch_param = None, device = "cuda"):
        super(base_model, self).__init__()
        self.device = device
        self.acts_per_episode = 12


        #VGG16 parameters
        #Note: one Fully connected layer was eliminated
        #self.channels = [64,64,128,128,256,256,256,512,512,512,512,512,512,512]
        if ch_param == None :
            self.ch_param = nn.Parameter( torch.Tensor([64,64,128,128,256,256,256,512,512,512,512,512,512]),requires_grad=False)
            self.channels = self.ch_param.data.int().tolist()
        else :
            self.ch_param = nn.Parameter(ch_param)
            self.channels = self.ch_param.data.int().tolist()

        self.linear_in = 512 
        self.convs = []
        self.relus = []
        self.pools = []
        self.convs.append( nn.Conv2d(3,self.channels[0],3,padding=1))
        self.relus.append(nn.ReLU())
        for i in range(0,len(self.channels)-1):
            self.convs.append(DynaConv2d(self.channels[i],self.channels[i+1],3,padding=1))
            self.relus.append(nn.ReLU())

        for i in range(5):
            self.pools.append(nn.MaxPool2d(2,stride=2))

        self.flatten = nn.Flatten()
        self.adaptive = nn.AdaptiveAvgPool2d((7,7))
        ## classifier's parameter
        self.classifier = nn.Sequential(
            nn.Linear(self.linear_in * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 10),
        )

    
        self.Agent = Agent(in_channels,num_actions,device= self.device)
        self.num_act = num_actions
        self.in_channels = in_channels
        self.convs = nn.ModuleList(self.convs)
        self.relus = nn.ModuleList(self.relus)
        self.pools = nn.ModuleList(self.pools)

        self._initialize_weights()
        self.actions= torch.zeros((self.acts_per_episode,))
        self.done = torch.zeros((self.acts_per_episode,))
        self.qvals = torch.zeros((self.acts_per_episode,num_actions))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, DynaConv2d) :
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def set_Agent_mode(self,flag):
        if flag == "reinforcement":
            self.Agent.is_training = True
            self.set_Agent_grad(True)
            self.Agent.train()
        elif flag == "supervised":
            self.Agent.is_training = False
            self.set_Agent_grad(False)
            self.Agent.eval()
        elif flag=="init_training" :
            self.Agent.is_training = True 
            self.Agent.epsilon = 1
            self.set_Agent_grad(False)
            self.Agent.eval()
        else :
            print("not defined options")
    
    def set_Agent_grad(self,requires_grad=False):
        for param in self.Agent.parameters():
            param.requires_grad = requires_grad



    def get_Agent_parameter(self):
        return self.Agent.state_dict()


    def load_Agent_parameter(self,path=None,target=None):
        if target ==None :
            self.Agent.load_state_dict(torch.load(path))
        else :
            self.Agent.load_state_dict(target)

    def _pre_encode(self,x):
        x = x.mean(dim=(-2,-1))
        x = x.view(x.shape[0],-1,1)
        return x
    
    def redistribute_act(self,thres=1e-4):
        i = 1
        for m in self.modules():
                if  isinstance(m,DynaConv2d):
                    nr = torch.linalg.norm(m.weight,dim=(1,2,3))
                    dis_b = nr.shape[0]
                    dis_s = nr.shape[0] - self.num_act
                    
                    while(True):
                        bool_nr = nr[dis_s:dis_b] <= thres 
                        print(bool_nr,print(nr[dis_s:dis_b]))
                        if torch.any(bool_nr) and self.channels[i] > self.num_act:
                            self.channels[i] -= self.num_act 
                            dis_b -= self.num_act
                            dis_s -= self.num_act
                        else :
                            break 
                    i += 1
        
        self.ch_param.data = torch.Tensor(self.channels)
        assert self.channels[0] ==64

                
                    

    def forward(self,x):
        x  =self.convs[0](x)
        x = self.relus[0](x)
        ## Agent action
       
        y = self._pre_encode(x)
        q_val = self.Agent(y)
        action = self.Agent.Action(q_val)
        #acts = (action+1) * torch.div(self.channels[0], self.num_act, rounding_mode='floor')
        acts = (action+1) * torch.div(self.channels[0], self.num_act, rounding_mode='floor')
        x ,x_len= self.convs[1](x,acts)
        x = self.relus[1](x)
        x = self.pools[0](x)
        ## Agent action
        
        y = self._pre_encode(x)
        q_val = self.Agent(y,x_len)
        action = self.Agent.Action(q_val)
        acts = (action+1) * torch.div(self.channels[1], self.num_act, rounding_mode='floor')
        x ,x_len= self.convs[2](x,acts)
        x = self.relus[2](x)
        ## Agent action
        
        y = self._pre_encode(x)
        q_val = self.Agent(y,x_len)
        action = self.Agent.Action(q_val)
        acts = (action+1) * torch.div(self.channels[2], self.num_act, rounding_mode='floor')
        x ,x_len= self.convs[3](x,acts)
        x = self.relus[3](x)
        x = self.pools[1](x)
        ## Agent action
        
        y = self._pre_encode(x)
        q_val = self.Agent(y,x_len)
        action = self.Agent.Action(q_val)
        acts = (action+1) * torch.div(self.channels[3], self.num_act, rounding_mode='floor')
        x ,x_len= self.convs[4](x,acts)
        x = self.relus[4](x)
        ## Agent action
        
        y = self._pre_encode(x)
        q_val = self.Agent(y,x_len)
        action = self.Agent.Action(q_val)
        acts = (action+1) * torch.div(self.channels[4], self.num_act, rounding_mode='floor')
        x ,x_len= self.convs[5](x,acts)
        x = self.relus[5](x)
        ## Agent action
        
        y = self._pre_encode(x)
        q_val = self.Agent(y,x_len)
        action = self.Agent.Action(q_val)
        acts = (action+1) * torch.div(self.channels[5], self.num_act, rounding_mode='floor')
        x ,x_len= self.convs[6](x,acts)
        x = self.relus[6](x)
        x = self.pools[2](x)
        ## Agent action
        
        y = self._pre_encode(x)
        q_val = self.Agent(y,x_len)
        action = self.Agent.Action(q_val)
        acts = (action+1) * torch.div(self.channels[6], self.num_act, rounding_mode='floor')
        x ,x_len= self.convs[7](x,acts)
        x = self.relus[7](x)
        ## Agent action
        
        y = self._pre_encode(x)
        q_val = self.Agent(y,x_len)
        action = self.Agent.Action(q_val)
        acts = (action+1) * torch.div(self.channels[7], self.num_act, rounding_mode='floor')
        x ,x_len= self.convs[8](x,acts)
        x = self.relus[8](x)
        ## Agent action
        
        y = self._pre_encode(x)
        q_val = self.Agent(y,x_len)
        action = self.Agent.Action(q_val)
        acts = (action+1) * torch.div(self.channels[8], self.num_act, rounding_mode='floor')
        x ,x_len= self.convs[9](x,acts)
        x = self.relus[9](x)
        x = self.pools[3](x)
        ## Agent action
        
        y = self._pre_encode(x)
        q_val = self.Agent(y,x_len)
        action = self.Agent.Action(q_val)
        acts = (action+1) * torch.div(self.channels[9], self.num_act, rounding_mode='floor')
        x ,x_len= self.convs[10](x,acts)
        x = self.relus[10](x)
        ## Agent action
        
        y = self._pre_encode(x)
        q_val = self.Agent(y,x_len)
        action = self.Agent.Action(q_val)
        acts = (action+1) * torch.div(self.channels[10], self.num_act, rounding_mode='floor')
        x ,x_len= self.convs[11](x,acts)
        x = self.relus[11](x)
        ## Agent action
        
        y = self._pre_encode(x)
        q_val = self.Agent(y,x_len)
        action = self.Agent.Action(q_val)
        acts = (action+1) * torch.div(self.channels[11], self.num_act, rounding_mode='floor')
        x ,x_len= self.convs[12](x,acts)
        x = self.relus[12](x)
        x = self.pools[4](x)

        bt, ch, h,w = x.shape
        zero_pad = torch.zeros((bt,self.linear_in-ch,h,w),device=self.device)
        x = torch.cat([x,zero_pad],dim=1)

        x = self.adaptive(x)
        x = self.flatten(x)
        x = self.classifier(x)

        return x

    @torch.no_grad()
    def forward_rl(self,x):
        states = []
        assert x.shape[0] == 1, "input shape for forward_rl invalid as : {0}".format(x.shape)

        x = self.convs[0](x)
        x = self.relus[0](x)
        ## Agent action
        pre_acts = torch.full((self.num_act,),x.shape[1])
        x_len = [x.shape[1] for i in range(x.shape[0])]
        y = self._pre_encode(x)

        q_val = self.Agent(y,x_len)
        action = self.Agent.Action(q_val)
        acts = (action+1) * torch.div(self.channels[0], self.num_act, rounding_mode='floor')
        #get action for replay buffer
        states.append(y)
        
        self.actions[0] = action
        self.done[0] = 0
        self.qvals[0] = q_val


        x ,x_len= self.convs[1](x,acts)
        x = self.relus[1](x)
        x = self.pools[0](x)
        ## Agent action
        
        y = self._pre_encode(x)
        q_val = self.Agent(y,x_len)
        action = self.Agent.Action(q_val)
        acts = (action+1) * torch.div(self.channels[1], self.num_act, rounding_mode='floor')
        #get action for replay buffer
        states.append(y)
        
        self.actions[1] = action
        self.done[1] = 0
        self.qvals[1] = q_val

        x ,x_len= self.convs[2](x,acts)
        x = self.relus[2](x)
        ## Agent action
        
        y = self._pre_encode(x)
        q_val = self.Agent(y,x_len)
        action = self.Agent.Action(q_val)
        acts = (action+1) * torch.div(self.channels[2], self.num_act, rounding_mode='floor')
        #get action for replay buffer
        states.append(y)
        
        self.actions[2] = action
        self.done[2] = 0
        self.qvals[2] = q_val

        x ,x_len= self.convs[3](x,acts)
        x = self.relus[3](x)
        x = self.pools[1](x)
        ## Agent action
        
        y = self._pre_encode(x)
        q_val = self.Agent(y,x_len)
        action = self.Agent.Action(q_val)
        acts = (action+1) * torch.div(self.channels[3], self.num_act, rounding_mode='floor')
        #get action for replay buffer
        states.append(y)
        
        self.actions[3] = action
        self.done[3] = 0
        self.qvals[3] = q_val

        x ,x_len= self.convs[4](x,acts)
        x = self.relus[4](x)
        ## Agent action
        
        y = self._pre_encode(x)
        q_val = self.Agent(y,x_len)
        action = self.Agent.Action(q_val)
        acts = (action+1) * torch.div(self.channels[4], self.num_act, rounding_mode='floor')
        #get action for replay buffer
        states.append(y)
        
        self.actions[4] = action
        self.done[4] = 0
        self.qvals[4] = q_val

        x ,x_len= self.convs[5](x,acts)
        x = self.relus[5](x)
        ## Agent action
        
        y = self._pre_encode(x)
        q_val = self.Agent(y,x_len)
        action = self.Agent.Action(q_val)
        acts = (action+1) * torch.div(self.channels[5], self.num_act, rounding_mode='floor')
        #get action for replay buffer
        states.append(y)
        
        self.actions[5] = action
        self.done[5] = 0
        self.qvals[5] = q_val

        x ,x_len= self.convs[6](x,acts)
        x = self.relus[6](x)
        x = self.pools[2](x)
        ## Agent action
        
        y = self._pre_encode(x)
        q_val = self.Agent(y,x_len)
        action = self.Agent.Action(q_val)
        acts = (action+1) * torch.div(self.channels[6], self.num_act, rounding_mode='floor')
        #get action for replay buffer
        states.append(y)
        
        self.actions[6] = action
        self.done[6] = 0
        self.qvals[6] = q_val

        x ,x_len= self.convs[7](x,acts)
        x = self.relus[7](x)
        ## Agent action
        
        y = self._pre_encode(x)
        q_val = self.Agent(y,x_len)
        action = self.Agent.Action(q_val)
        acts = (action+1) * torch.div(self.channels[7], self.num_act, rounding_mode='floor')
        #get action for replay buffer
        states.append(y)
        
        self.actions[7] = action
        self.done[7] = 0
        self.qvals[7] = q_val

        x ,x_len= self.convs[8](x,acts)
        x = self.relus[8](x)
        ## Agent action
        
        y = self._pre_encode(x)
        q_val = self.Agent(y,x_len)
        action = self.Agent.Action(q_val)
        acts = (action+1) * torch.div(self.channels[8], self.num_act, rounding_mode='floor')
        #get action for replay buffer
        states.append(y)
        
        self.actions[8] = action
        self.done[8] = 0
        self.qvals[8] = q_val

        x ,x_len= self.convs[9](x,acts)
        x = self.relus[9](x)
        x = self.pools[3](x)
        ## Agent action
        
        y = self._pre_encode(x)
        q_val = self.Agent(y,x_len)
        action = self.Agent.Action(q_val)
        acts = (action+1) * torch.div(self.channels[9], self.num_act, rounding_mode='floor')
        #get action for replay buffer
        states.append(y)
        
        self.actions[9] = action
        self.done[9] = 0
        self.qvals[9] = q_val

        x ,x_len= self.convs[10](x,acts)
        x = self.relus[10](x)
        ## Agent action
        
        y = self._pre_encode(x)
        q_val = self.Agent(y,x_len)
        action = self.Agent.Action(q_val)
        acts = (action+1) * torch.div(self.channels[10], self.num_act, rounding_mode='floor')
        #get action for replay buffer
        states.append(y)
        
        self.actions[10] = action
        self.done[10] = 0
        self.qvals[10] = q_val

        x ,x_len= self.convs[11](x,acts)
        x = self.relus[11](x)
        ## Agent action
        
        y = self._pre_encode(x)
        q_val = self.Agent(y,x_len)
        action = self.Agent.Action(q_val)
        acts = (action+1) * torch.div(self.channels[11], self.num_act, rounding_mode='floor')
        #get action for replay buffer
        states.append(y)
        
        self.actions[11] = action
        self.done[11] = 1
        self.qvals[11] = q_val

        x ,x_len= self.convs[12](x,acts)
        x = self.relus[12](x)
        x = self.pools[4](x)

        bt, ch, h,w = x.shape
        zero_pad = torch.zeros((bt,self.linear_in-ch,h,w),device=self.device)
        x = torch.cat([x,zero_pad],dim=1)

        x = self.adaptive(x)
        x = self.flatten(x)
        x = self.classifier(x)

        return x, states, self.actions, self.done, self.qvals 
