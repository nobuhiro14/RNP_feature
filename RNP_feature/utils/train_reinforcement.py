import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import sys
sys.path.append('../')

from models.Agent import Agent
from models.replay_buffer import *
from utils.reward import *
import sys
import numpy as np
import os
import time
import csv 



def train_reinforcement(model,
                        Q_target,
                        epsilon,
                        optimizer_spec,
                        Rewards,
                        acts_per_episode,
                        model_num,
                        in_channels=1024,
                        num_actions = 4,
                        replay_buffer_size=10000,
                        batch_size=64,
                        learning_freq = 100,
                        learning_starts = 100,
                        target_update_freq = 50,
                        activate=False,
                        path = "results",
                        device ="cuda"):


    # initialize optimizer
    optimizer = optimizer_spec.constructor(model.Agent.parameters(), **optimizer_spec.kwargs)

    Q_target.eval()
    model.Agent.train()

    #initialize replay buffer
    frame_history_len = 1
    replay_buffer = ReplayBuffer(replay_buffer_size,frame_history_len,acts_per_episode)


    criterion = nn.CrossEntropyLoss()
    #### generate the dataset loader ####
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
        , batch_size = 1
        , shuffle = True)


    ###############
    # RUN ENV     #
    ###############
    num_param_updates = 0
    mean_episode_reward      = 0
    best_mean_episode_reward = 0
    LOG_EVERY_N_STEPS = 500
    SAVE_MODEL_EVERY_N_STEPS = 3000000
    LOWEST_EPSILON = 0.01
    EPSILON_DECAY = 50000
    model.Agent._decay_epsilon(epsilon,LOWEST_EPSILON,EPSILON_DECAY)
    mean_episode_reward_tmp = 0
    loss_ave = 0
    ave_corrects = 0
    error_ave = 0

    log_loss_name = f"{path}/loss_log_No{model_num}.csv"
    log_rew_name = f"{path}/reward_log_No{model_num}.csv"
    log_Q_name = f"{path}/qvalue_log_No{model_num}.csv"
    log_mean_loss_name = f"{path}/mean_loss_log_No{model_num}.csv"
    f_qval = open(log_Q_name,'w')
    f_loss = open(log_loss_name,'w') 
    f=  open(log_rew_name,'w') 
    f_mean_loss = open(log_mean_loss_name,"w")
    meanwriter = csv.writer(f_mean_loss)
    writer = csv.writer(f)
    lswriter = csv.writer(f_loss)
    qvalwriter = csv.writer(f_qval)


    for epoch , (imgs, labels) in enumerate(train_loader):



        imgs = imgs.to(device)
        labels = labels.to(device)

        labels_hat,states_ep, actions_ep, done_ep,qval_ep = model.forward_rl(imgs)
        n_corrects = (labels_hat.argmax(axis=1)==labels).sum().item()
        loss_value = criterion(labels_hat,labels)

        if activate :
            model.Agent._decay_epsilon()

        actions_ep = actions_ep.to("cpu")
        done_ep = done_ep.to("cpu")
        ls_cpu = loss_value.to("cpu")
        rewards_ep = Rewards.get_reward(actions_ep,done_ep,ls_cpu)
        assert rewards_ep[0] <=0, "reward get positive {0}".format(rewards_ep)

        writer.writerow([sum(rewards_ep)])

        labels_hat = labels_hat.to("cpu")
        states_ep = states_ep
        
        replay_buffer.store_frame(states_ep,actions_ep,rewards_ep,done_ep)
        ###################### start training #################
        if (epoch > learning_starts and
                epoch % learning_freq == 0 and
                replay_buffer.can_sample(batch_size)):

            loss_ave += loss_value
            ave_corrects += 100*(n_corrects/labels.size(0))

            # sample transition batch from replay memory
            # done_mask = 1 if next state is end of episode
            obs_t,xlen_t, act_t, rew_t, obs_tp1, xlen_tp1, done_mask = replay_buffer.sample(batch_size)
            obs_t = Variable(obs_t)
            act_t = Variable(act_t)
            rew_t = Variable(rew_t)
            obs_tp1 = Variable(obs_tp1)
            done_mask = Variable(done_mask)

            tmp = obs_t.to(device)
            assert tmp.shape[0] == 128 and tmp.shape[2] == 1, "invalid shape  {0}   ".format(tmp.shape)
            assert len(xlen_t) == 128 , "invalid shape xlen {0}    {1}".format(len(xlen_t), type(xlen_t))
            assert len(xlen_tp1) == 128 , "invalid shape xlen {0}    {1}".format(len(xlen_tp1), type(xlen_tp1))
            q_values = model.Agent(tmp)
            q_values = q_values.to("cpu")
            act_t  =act_t.to(torch.int64)
            q_values = q_values.squeeze()
            q_s_a = q_values.gather(1, act_t.unsqueeze(1))
            q_s_a = q_s_a.squeeze()

            tmp = obs_tp1.to(device)
            q_tp1_values = Q_target(tmp).detach()
            q_tp1_values = q_tp1_values.to("cpu")
            q_tp1_values.squeeze()
            q_s_a_prime, a_prime = q_tp1_values.max(-1)
            q_s_a_prime = q_s_a_prime.squeeze()
            # if current state is end of episode, then there is no next Q value
            q_s_a_prime = (1 - done_mask) * q_s_a_prime

            # Compute Bellman error
            # r + gamma * Q(s',a', theta_i_frozen) - Q(s, a, theta_i)
            gamma = torch.tensor(0.95)

            error = (rew_t + gamma * q_s_a_prime - q_s_a)**2  ## error with RNP article
            
            #error = (-1.0*rew_t + gamma * q_s_a_prime - q_s_a) 

            # clip the error and flip
            #clipped_error = -1.0 * error.clamp(-1, 1)
            #clipped_error = -1.0 * error
            clipped_error = error.clamp(-1,1)
            ls_write = sum(error)/len(error)
            ls_write = ls_write.to("cpu").detach().numpy()
            lswriter.writerow([ls_write])

            qvalwriter.writerows(qval_ep.detach().numpy())

            # backwards pass
            optimizer.zero_grad()
            q_s_a.backward(clipped_error)

            # update
            optimizer.step()
            num_param_updates += 1

            mean_episode_reward_tmp += sum(rewards_ep)/len(rewards_ep)

            error_ave += sum(error)/len(error)

            if epoch %100 == 0:
                mean_episode_reward = mean_episode_reward_tmp /100
                mean_episode_reward_tmp = 0
                loss_ave_show = loss_ave /100
                loss_ave = 0
                ave_corrects_show = ave_corrects/100
                ave_corrects = 0

            # update target Q network weights with current Q network weights
            if num_param_updates % target_update_freq == 0:
                print("############# updated ######################")
                Q_target.load_state_dict(model.get_Agent_parameter())




            if num_param_updates % SAVE_MODEL_EVERY_N_STEPS == 0:

                os.makedirs(path,exist_ok =True)
                model_save_path = f"{path}/base_model_%s.model" %( epoch)
                torch.save(model.state_dict(), model_save_path)
                model_save_path = f"{path}/Agent_%s.model" %( epoch)
                torch.save(model.Agent.state_dict(), model_save_path)


            if epoch % LOG_EVERY_N_STEPS == 0:
                error_ave = error_ave / LOG_EVERY_N_STEPS

                print("---------------------------------")
                print("Timestep %d" % (epoch,))
                print("learning started? %d" % (epoch > learning_starts))
                print("mean reward (100 episodes) %f" % mean_episode_reward)
                print("mean loss (100 episodes) %f" % loss_ave_show)
                print(f"mean error ({LOG_EVERY_N_STEPS} episodes) {error_ave}")
                print(f"ACC { ave_corrects_show}")
                print(f"alpha  :  {Rewards.alpha}")
                print("epsilon %f" % model.Agent.epsilon)
                print("learning_rate %f" % optimizer_spec.kwargs['lr'])
                print(act_t)
                print(rewards_ep)
                print(actions_ep)
                meanwriter.writerow([error_ave.to("cpu").detach().numpy()])
                error_ave = 0
                sys.stdout.flush()
                
    f.close()
    f_loss.close()
    f_qval.close()
    f_mean_loss.close()


    return model, Q_target
