import argparse
import os
import torch


class Options():

    def __init__(self):
        parser = argparse.ArgumentParser()
        ############### parameters of supervised learning ####################################
        parser.add_argument("--epoch",type = int, default = 30, help="epoch of supervised learning for base_model")
        parser.add_argument("--bt_size_sp", type=int, default=64, help ="batch size of supervised learning for base_model")
        parser.add_argument("--lr_sp", type=float, default=0.01, help =" learning rate of SGD (supervised)")
        parser.add_argument("--momentum", type=float, default=0.9, help="momentum for SGD (supervised)")
        parser.add_argument("--num_patience", type=int, default=3, help="Number of the model to store early stops")
        parser.add_argument("--lasso_flag", type = int,  default  = 1 , help="0: no regularization, 1: use group lasso, 2: use L1 norm 3: use L1 norm and Group Lasso")
        parser.add_argument("--lasso_param", type=float, default=9e-6, help="parameter of lasso")

        ############### parameters of reinforcement learning ####################################
        parser.add_argument("--lr_re", type=float, default=1e-6, help =" learning rate of RMSprop (reinforcement)")
        parser.add_argument("--p_ls", type = float, default=0.1, help = "reward for loss (reinforcement)")
        parser.add_argument("--bt_size_re", type=int, default=128, help ="batch size of reinforcement learning for agent")
        parser.add_argument("--lr_freq_re", type=int, default=50, help="frequency of the reinforcement learning ")
        parser.add_argument("--lr_start_re", type=int, default=512, help="starting epochs for reinforcement learning")
        parser.add_argument("--update_re", type=int, default=50, help="frequency of the update to target Agent")
        parser.add_argument("--in_channels",type=int, default=64, help="parameter for RNN at Agent")
        parser.add_argument("--num_actions",type=int, default = 4, help="devide the filters and create num_actions group at Convolution.")
        parser.add_argument("--buffer_size", type=int, default= 15000, help ="replay buffer size (reinforcement")
        parser.add_argument("--eps_re", type=float, default = 0.1, help="epsilon for RMSprop (reinforcement)")
        parser.add_argument("--alpha_re", type=float, default = 0.95, help="alpha for RMSprop (reinforcement)")
        parser.add_argument("--epsilon_re",type=float, default = 1, help="initial parameter of epsilon-greedy methods")
        parser.add_argument("--activate_decay",action = "store_true", help="whether epsilon greedy methods use decay epsilon, if it used, epsilon decays")


        ############### parameter of general learning ####################################
        parser.add_argument("--train_num",type=int, default = 3, help="training number for reinforcement+ supervised")
        parser.add_argument("--init_training",type=int, default = 0, help="0 : initial train 1: load created model 2: load_vgg model")
        parser.add_argument("--save_pth", type =str , default= "results", help="path of trained models")
        self.parser = parser


    def get_args(self):
        return self.parser.parse_args()


class TestOptions():

    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--load_pth", type =str , default= "results", help="path of trained models")
        parser.add_argument("--use_profiler",action = "store_true", help="it activate profiler")
        parser.add_argument("--load_vgg16",action = "store_true", help="it activate profiler")

        self.parser = parser


    def get_args(self):
        return self.parser.parse_args()