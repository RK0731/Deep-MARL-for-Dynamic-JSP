import random
import numpy as np
import sys
from pathlib import Path
import copy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from tabulate import tabulate

'''
The deep MARL learning/training algorithm
'''

class brain:
    def __init__(self, env, job_creator, machines, warm_up, span, *args, **kwargs):
        # initialize the environment and the machines to be controlled
        self.env = env
        self.job_creator = job_creator
        # training duration
        self.warm_up = warm_up
        self.span = span
        # m_list contains all machines on shop floor, we need them to collect data
        self.m_list = machines
        print(machines)
        self.m_no = len(self.m_list)
        # and build dicts that equals number of machines to be controlled in job creator
        self.job_creator.build_sqc_experience_repository(self.m_list)
        # activate the sequencing learning event of machines so they will collect data
        # and build dictionary to store the data
        print("+++ Take over all machines, activate learning mode +++")
        for m in self.m_list:
            m.sequencing_learning_event.succeed()
        '''
        choose the reward function for machines
        '''
        if 'reward_function' in kwargs:
            order = 'm.reward_function = m.get_reward{}'.format(kwargs['reward_function'])
            for m in self.m_list:
                exec(order)
        else:
            print('WARNING: reward function is not specified')
            raise Exception
        '''
        chooose the architecture of DRL, then state and action funciton is determined accordlingly
        and specify the path to store the trained state-dict
        needs to be specified in kwargs, otherwise abstract networks + abstract state space
        there is an action_NN that perform the actual action and be trained
        and a target_NN to improve the stability of training
        '''

        if 'TEST' in kwargs and kwargs['TEST']:
            print("---!!! TEST mode ON !!!---")
            self.input_size =  self.state_direct(self.m_list[0].sequencing_data_generation()).size()
            self.input_size_as_list = list(self.input_size)
            self.output_size = 4
            self.action_NN = network_TEST(self.input_size, self.output_size)
            self.target_NN = copy.deepcopy(self.action_NN)
            self.model_path = Path.cwd()/"trained_models"/f"TEST_DDQN_rwd{kwargs['reward_function']}.pt"
            self.build_state = self.state_direct
            #self.train = self.train_prioritized_DDQN
            self.train = self.train_Double_DQN
            self.action_DRL = self.action_direct
            for m in self.m_list:
                m.build_state = self.state_direct
        else: # the default mode is DDQN
            print("---X DEFAULT (DDQN) mode ON X---")
            self.input_size =  self.state_direct(self.m_list[0].sequencing_data_generation()).size()
            self.input_size_as_list = list(self.input_size)
            self.output_size = 4
            self.action_NN = network_value_based(self.input_size, self.output_size)
            self.target_NN = copy.deepcopy(self.action_NN)
            self.model_path = Path.cwd()/"trained_models"/f"DDQN_rwd{kwargs['reward_function']}.pt"
            self.build_state = self.state_direct
            self.train = self.train_Double_DQN
            self.action_DRL = self.action_direct
            for m in self.m_list:
                m.build_state = self.state_direct

        '''
        sometimes train based on trained parameters can save time
        importing trained parameters from specified path
        '''
        if kwargs['bsf_start']: # import best for far trained parameters to kick off
            if kwargs['TEST']:
                import_path = Path.cwd()/"trained_models"/"bsf_TEST.pt"
            else:
                import_path = Path.cwd()/"trained_models"/"bsf_DDQN.pt"
            self.action_NN.network.load_state_dict(torch.load(import_path))
            print("IMPORT FROM:", import_path)
        '''
        new path for storing the trained parameters, if specified
        '''
        if 'store_to' in kwargs:
            self.model_path = Path.cwd()/f"trained_models{kwargs['store_to']}.pt"
            print("New path:", self.model_path)
        '''
        initialize all training parameters by default value
        '''
        # initialize initial replay memory and TD error
        self.rep_memo = []
        self.rep_memo_TDerror = []
        # some training parameters
        self.minibatch_size = 64
        self.rep_memo_size = 1024
        self.action_NN_training_interval = 5 # training frequency of updating the action network
        self.action_NN_training_time_record = []
        self.target_NN_sync_interval = 250  # synchronize the weights of NN every xx time units
        self.target_NN_sync_time_record = []
        # Initialize the parameters for learning of DRL
        self.discount_factor = 0.95 # how much agent care about long-term rewards
        self.epsilon = 0.4  # chance of exploration
        # record the training
        self.loss_time_record = []
        self.loss_record = []
        # warmup process
        self.env.process(self.warm_up_process())
        for m in self.m_list:
            m.job_sequencing = self.random_exploration
        # training processes
        self.env.process(self.training_process_parameter_sharing())
        self.env.process(self.update_rep_memo_parameter_sharing_process())
        self.build_initial_rep_memo = self.build_initial_rep_memo_parameter_sharing
        '''
        these two processes are shared among all schemes
        '''
        self.env.process(self.sync_network_process())
        self.env.process(self.update_training_parameters_process())
        for x in self.action_NN.parameters():
            print(np.prod(list(x.shape)))


    '''
    1. downwards for functions that required for the simulation
       including the warm-up, action functions and multiple sequencing rules
       those functions are also used by validation module
    '''

    def warm_up_process(self): # warm up with random exploration
        print("random exploration from time {} to time {}".format(self.env.now, self.warm_up))
        yield self.env.timeout(self.warm_up - 1)
        # after the warm up period, build replay memory and start training
        self.build_initial_rep_memo()
        # hand over the target machines' sequencing function to DRL (action network)
        for m in self.m_list:
            m.job_sequencing = self.action_DRL

    def random_exploration(self, sqc_data):
        s_t = self.build_state(sqc_data)
        #print('state:',self.env.now,s_t)
        # action is a random index of job in action space
        a_t = torch.randint(self.output_size,[])
        self.strategic_idleness_bool = False # then no need do strategic idleness
        job_position, j_idx = self.action_conversion(a_t)
        m_idx = sqc_data[-1]
        # add the state at sequencing to job creator's corresponding repository
        self.build_experience(j_idx,m_idx,s_t,a_t)
        return job_position, self.strategic_idleness_bool, self.strategic_idleness_time

    '''action function for actual DRL control'''
    def action_direct(self, sqc_data): # strategic idleness is prohibitted
        s_t = self.build_state(sqc_data)
        m_idx = sqc_data[-1]
        # Epsilon-greedy policy
        if random.random() < self.epsilon:
            a_t = torch.randint(0,self.output_size,[])
            #print('Random Selection:', a_t)
        else:
            # input state to action network, produce the state-action value
            value = self.action_NN.forward(s_t.reshape([1]+self.input_size_as_list),m_idx).squeeze()
            # greedy policy
            a_t = torch.argmax(value)
            #print("State is:", s_t)
            #print('State-Action Values:', value)
            #print('Direct Selection: %s'%(a_t))
        self.strategic_idleness_bool = False # then no need do strategic idleness
        job_position, j_idx = self.action_conversion(a_t)
        # add the state at sequencing to job creator's corresponding repository
        self.build_experience(j_idx,m_idx,s_t,a_t)
        return job_position, self.strategic_idleness_bool, self.strategic_idleness_time


    '''
    2. downwards are functions used for building the state of the experience (replay memory)
       those functions are also used by validation module

    state functions under this category will be called twice in each operation
    before the execution of operation (s_t), and after (s_t+1)
    '''


    def state_direct(self, sqc_data): # presenting information of job which satisfies certain criteria
        '''STEP 1: check queuing jobs, if any, clip the sqc data'''
        # number of candidate jobs
        no_candidate_jobs = len(sqc_data[0])
        if no_candidate_jobs == 1: # if there's only one queuing job, simply copy the info of the only job (most common case)
            # original sqc_data contains lots of things that won't be used, create a clipped copy of it
            clipped_data = np.concatenate([sqc_data[0], sqc_data[1], sqc_data[5], sqc_data[7], sqc_data[10]])
            s_t = [clipped_data for i in range(4)]
            # and set the correspondence to the first job in the queue
            self.correspondence_pos = [0 for i in range(4)]
            self.correspondence_idx = [0 for i in range(4)]
        elif no_candidate_jobs == 0 : # if there's no queuing job, create dummy state that all enrties are 0
            s_t = [np.array([0 for i in range(5)]) for i in range(4)]
            # and set the correspondence to dummy value
            self.correspondence_pos = [-1 for i in range(4)]
            self.correspondence_idx = [-1 for i in range(4)]
        else: # if there's multiple jobs, try include them exhaustively
            # empty list of position and index of candidate jobs
            s_t = [] # initialize empty state
            self.correspondence_pos = []
            self.correspondence_idx = []
            clipped_data = np.array([sqc_data[0], sqc_data[1], sqc_data[5], sqc_data[7], sqc_data[10]])
            # copy the lists for exhaustive inclusion
            copied_clipped_data = clipped_data.copy() # jobs would be gradually kicked out from a copy of clipped_data
            exhaust_idx = sqc_data[-2].copy() # also kick out from list of indexes
            exhaust_pos = np.arange(no_candidate_jobs) # also kick out from list of position
            row_number = [0,1,2,3] # spt, lwkr, ms, avlm
            row = 0
            # first try to include all jobs, reduce duplication as possible
            try:
                for i in range(4):
                    #print(copied_clipped_data, exhaust_idx, exhaust_pos, self.env.now)
                    no_duplication_pos = np.argmin(copied_clipped_data[row])
                    job_idx = exhaust_idx[no_duplication_pos]
                    job_pos = exhaust_pos[no_duplication_pos]
                    self.correspondence_idx.append(job_idx)
                    self.correspondence_pos.append(job_pos)
                    s_t.append(copied_clipped_data[:,no_duplication_pos])
                    row += 1
                    # kick out the selected job from exhaust list
                    copied_clipped_data = np.delete(copied_clipped_data, no_duplication_pos, axis=1)
                    exhaust_idx = np.delete(exhaust_idx, no_duplication_pos)
                    exhaust_pos = np.delete(exhaust_pos, no_duplication_pos)
            # if number of candidate job less than 4 (expection raise), then return to normal procedure to complete the state
            except:
                for i in range(row,4):
                    normal_pos = np.argmin(clipped_data[row])
                    normal_idx = sqc_data[-2][normal_pos]
                    self.correspondence_idx.append(normal_idx)
                    self.correspondence_pos.append(normal_pos)
                    s_t.append(clipped_data[:,normal_pos])
                    row += 1
        '''STEP 2: get information of arriving jobs and others'''
        arriving_jobs = np.where(self.job_creator.next_machine_list == sqc_data[-1])[0] # see if there are jobs will arrive
        self.arriving_job_exists = bool(len(arriving_jobs)) # get the bool variable to represent whether arriving job exists
        # get the available time of machine itself
        avlm_self = self.job_creator.available_time_list[sqc_data[-1]] - self.env.now
        #print(self.job_creator.next_machine_list, self.job_creator.release_time_list, self.env.now)
        #print('%s arriving jobs from machine %s'%(self.arriving_job_exists,arriving_jobs))
        if self.arriving_job_exists: # if there are arriving jobs
            # get the exact next job that will arrive at machine out of all arriving jobs
            pos = arriving_jobs[self.job_creator.release_time_list[arriving_jobs].argmin()]
            arriving_j_idx = self.job_creator.current_j_idx_list[pos]
            # and retrive the information of this job
            pt_self = self.job_creator.next_pt_list[pos]
            rem_pt = self.job_creator.arriving_job_rempt_list[pos]
            slack = self.job_creator.arriving_job_slack_list[pos]
            self.strategic_idleness_time = self.job_creator.release_time_list[arriving_jobs].min() - self.env.now # how long to wait if agent decide to wait for arriving job
            arriving_job_info = np.array([pt_self, rem_pt, slack, avlm_self, self.strategic_idleness_time])
        else: # if there is no arriving job
            arriving_j_idx = None
            arriving_job_info = np.array([0, 0, 0, avlm_self, 0])
            self.strategic_idleness_time = 0 # no need to wait for any arriving jobs
        # add position and index of arriving job to correspondence
        self.correspondence_pos.append(len(sqc_data[0]))
        self.correspondence_idx.append(arriving_j_idx)
        s_t.append(arriving_job_info)
        '''STEP 3: finally, convert list to tensor and output it'''
        s_t = torch.FloatTensor(s_t)
        #print('state:',s_t)
        return s_t

    # convert the action to the position of job in queue, and to the index of job so the job can be picked and recorded
    def action_conversion(self, a_t):
        #print(a_t)
        job_position = self.correspondence_pos[a_t]
        j_idx = self.correspondence_idx[a_t]
        #print(self.correspondence_idx)
        #print(self.correspondence_pos)
        #print('selected job idx: %s, position in queue: %s'%(j_idx, job_position))
        return job_position, j_idx

    # add the experience to job creator's incomplete experiece memory
    def build_experience(self,j_idx,m_idx,s_t,a_t):
        self.job_creator.incomplete_rep_memo[m_idx][self.env.now] = [s_t,a_t]


    '''
    3. downwards are functions used for building / updating replay memory
    '''


    # called after the warm-up period
    def build_initial_rep_memo_parameter_sharing(self):
        #print(self.job_creator.rep_memo)
        for m in self.m_list:
            # copy the initial memoery from corresponding rep_memo from job creator
            #print('%s complete and %s incomplete experience for machine %s'%(len(self.job_creator.rep_memo[m.m_idx]), len(self.job_creator.incomplete_rep_memo[m.m_idx]), m.m_idx))
            #print(self.job_creator.incomplete_rep_memo[m.m_idx])
            self.rep_memo += self.job_creator.rep_memo[m.m_idx].copy()
            # and clear the replay memory in job creator, keep it updated
            self.job_creator.rep_memo[m.m_idx] = []
        # and the initial dummy TDerror
        self.rep_memo_TDerror = torch.ones(len(self.rep_memo),dtype=torch.float)
        print('INITIALIZATION - replay_memory')
        print(tabulate(self.rep_memo, headers = ['s_t','a_t','s_t+1','r_t']))
        print('INITIALIZATION - size of replay memory:',len(self.rep_memo))
        print('---------------------------initialization accomplished-----------------------------')

    # update the replay memory periodically
    def update_rep_memo_parameter_sharing_process(self):
        yield self.env.timeout(self.warm_up)
        while self.env.now < self.span:
            for m in self.m_list:
                # add new memoery from corresponding rep_memo from job creator
                self.rep_memo += self.job_creator.rep_memo[m.m_idx].copy()
                # and assign top priority to new experiences
                self.rep_memo_TDerror = torch.cat([self.rep_memo_TDerror, torch.ones(len(self.job_creator.rep_memo[m.m_idx]),dtype=torch.float)])
                # and clear the replay memory in job creator, keep it updated
                self.job_creator.rep_memo[m.m_idx] = []
            # clear the obsolete experience periodically
            if len(self.rep_memo) > self.rep_memo_size:
                truncation = len(self.rep_memo)-self.rep_memo_size
                self.rep_memo = self.rep_memo[truncation:]
                self.rep_memo_TDerror = self.rep_memo_TDerror[truncation:]
            #print(self.rep_memo_TDerror)
            yield self.env.timeout(self.action_NN_training_interval*10)


    '''
    4. downwards are functions used in the training
       such as the main training process, and training parameters update
    '''


    # parameter sharing mode is on
    def training_process_parameter_sharing(self):
        # wait for the warm up
        yield self.env.timeout(self.warm_up)
        # pre-train the policy NN before hand over to it
        for i in range(10):
            self.train()
        # periodic training
        while self.env.now < self.span:
            self.train()
            yield self.env.timeout(self.action_NN_training_interval)
        # end the training after span time
        # and store the trained parameters
        print('FINAL- replay_memory')
        print(tabulate(self.rep_memo, headers = ['s_t','a_t','s_t+1','r_t']))
        print('FINAL - size of replay memory:',len(self.rep_memo))
        # save the parameters of policy / action network after training
        torch.save(self.action_NN.network.state_dict(), self.model_path)
        # after the training, print out the setting of DRL architecture
        print("Training terminated, store trained parameters to: {}".format(self.model_path))

    # synchronize the ANN and TNN, and some settings
    def sync_network_process(self):
        # one second after the initial training, so we can have a slightly better target network
        yield self.env.timeout(self.warm_up+1)
        while self.env.now < self.span:
            # synchronize the parameter of policy and target network
            self.target_NN = copy.deepcopy(self.action_NN)
            print('--------------------------------------------------------')
            print('the target network and epsilion are updated at time %s' % self.env.now)
            print('--------------------------------------------------------')
            yield self.env.timeout(self.target_NN_sync_interval)

    # reduce the learning rate periodically
    def update_training_parameters_process(self):
        # one second after the initial training
        yield self.env.timeout(self.warm_up)
        reduction = (self.action_NN.lr - self.action_NN.lr/10)/10
        while self.env.now < self.span:
            yield self.env.timeout((self.span-self.warm_up)/10)
            # reduce the learning rate
            self.action_NN.lr -= reduction
            self.epsilon -= 0.002
            print('--------------------------------------------------------')
            print('learning rate adjusted to {} at time {}'.format(self.action_NN.lr, self.env.now))
            print('--------------------------------------------------------')

    def train_Double_DQN(self):
        """
        draw the random minibatch to train the network
        every element in the replay menory is a list [s_0, a_0, s_1, r_0]
        all element of this list are tensors
        """
        size = min(len(self.rep_memo),self.minibatch_size)
        minibatch = random.sample(self.rep_memo,size)
        '''
        slice, and stack the 1D tensors to several 3D tensors (batch, channel, vector)
        the "torch.stack" is only applicable when the augment is a list of tensors, or multi-dimensional tensor
        '''
        sample_s0_batch = torch.stack([data[0] for data in minibatch], dim=0).reshape([size]+self.input_size_as_list)
        sample_s1_batch = torch.stack([data[2] for data in minibatch], dim=0).reshape([size]+self.input_size_as_list)
        sample_a0_batch = torch.stack([data[1] for data in minibatch], dim=0).reshape(size,1)
        sample_r0_batch = torch.stack([data[3] for data in minibatch], dim=0).reshape(size,1)
        '''
        the size of these batches:
        sample_s0_batch = sample_s1_batch = minibatch size * 1 * input_size
        sample_a0_batch = sample_r0_batch = minibatch size * m_no
        sample_r0_batch = minibatch size
        '''
        # get the Q value (current value of state-action pair) of s0
        Q_0 = self.action_NN.forward(sample_s0_batch)
        #print('Q_0 is:\n', Q_0)
        #print('a_0 is:', sample_a0_batch)
        # get the current state-action value of actions that would have been taken
        current_value = Q_0.gather(1, sample_a0_batch)
        #print('current value is:', current_value)
        '''
        get the Q Value of s_1 in both action and target network, to estimate the state value
        architecture is DDQN, NOT DQN !!!
        evaluate the greedy policy according to action network, but using the target network to estimate the value
        '''
        Q_1_action = self.action_NN.forward(sample_s1_batch).detach()
        Q_1_target = self.target_NN.forward(sample_s1_batch).detach()
        #print('Q_1_action is:\n', Q_1_action)
        #print('Q_1_target is:\n', Q_1_target)
        '''
        size of Q_0, Q_1_action and Q_1_target = minibatch size * m_no
        they're 2D tensors
        '''
        max_Q_1_action, max_Q_1_action_idx = torch.max(Q_1_action, dim=1) # use action network to get action, rather than max operation
        #print('max value of Q_1_action is:\n', max_Q_1_action)
        max_Q_1_action_idx = max_Q_1_action_idx.reshape([size,1])
        #print('max idx of Q_1_action is:\n', max_Q_1_action_idx)
        # adjust the max_Q of s_0 by the discount factor (refer to Bellman Equation and TD)
        next_state_value = Q_1_target.gather(1, max_Q_1_action_idx)
        #print('estimated value of next state is:\n', next_state_value)
        next_state_value *= self.discount_factor
        #print('discounted next state value is:\n', next_state_value)
        '''
        the sum of reward and discounted max_Q is the target value
        target value is 2D matrix, size = minibatch_size * m_no
        '''
        #print('reward batch is:', sample_r0_batch)
        target_value = (sample_r0_batch + next_state_value)
        #print('target value is:', target_value)
        #print('TD error:',target_value - current_value)
        # calculate the loss
        loss = self.action_NN.loss_func(current_value, target_value)
        self.loss_time_record.append(self.env.now)
        self.loss_record.append(float(loss))
        if not self.env.now%50:
            print('Time: %s, loss: %s:'%(self.env.now, loss))
        # first, clear the gradient (old) of parameters
        self.action_NN.optimizer.zero_grad()
        # second, calculate gradient (new) of parameters
        loss.backward(retain_graph=True)
        '''
        # check the gradient, to avoid exploding/vanishing gradient, very seldom though
        for param in self.action_NN.module_dict[m_idx].parameters():
            print(param.grad.norm())
        '''
        # third, update the parameters
        self.action_NN.optimizer.step()

    # print out the functions and classes used in the training
    def check_parameter(self):
        print('-------------  Training Setting Check  -------------')
        print("Address seed:",self.model_path)
        print('Rwd.Func.:',self.m_list[0].reward_function.__name__)
        print('State Func.:',self.build_state.__name__)
        print('Action Func.:',self.action_DRL.__name__)
        print('Training Func.:',self.train.__name__)
        print('ANN:',self.action_NN.__class__.__name__)
        print('------------- Training Parameter Check -------------')
        print('Discount rate:',self.discount_factor)
        print('Train feq: %s, Sync feq: %s'%(self.action_NN_training_interval,self.target_NN_sync_interval))
        print('Rep memo: %s, Minibatch: %s'%(self.rep_memo_size,self.minibatch_size))
        print('------------- Training Scenarios Check -------------')
        print("PT heterogeneity:",self.job_creator.pt_range)
        print('Due date tightness:',self.job_creator.tightness)
        print('Utilization rate:',self.job_creator.E_utliz)
        print('----------------------------------------------------')


'''
class of sequencing functions to kick-off the training (optional)
'''
class sqc_func:
    def PTWINQS(s_t):
        data = s_t[:4,:]
        sum = data[:,0] + data[:,2] + data[:,3]
        a_t = sum.argmin()
        #print(s_t, data,sum,a_t)
        return a_t

    def DPTLWKRS(s_t):
        data = s_t[:4,:]
        sum = data[:,0] + data[:,1] + data[:,2]
        a_t = sum.argmin()
        return a_t

    def MDD(s_t):
        data = s_t[:4,:]
        due = data[:,1] + data[:,2]
        finish = data[:,1]
        #print(due,finish)
        MDD, MDD_idx = torch.stack([due,finish],dim=0).max(dim=0)
        a_t = MDD.argmin()
        return a_t

    def SPT(s_t):
        data = s_t[:4,:]
        pt = data[:,0]
        a_t = pt.argmin()
        return a_t

    def WINQ(s_t):
        data = s_t[:4,:]
        sum = data[:,3]
        a_t = sum.argmin()
        return a_t

    def MS(s_t):
        data = s_t[:4,:]
        sum = data[:,2]
        a_t = sum.argmin()
        return a_t

    def CR(s_t):
        data = s_t[:4,:]
        sum = data[:,2]/data[:,1]
        a_t = sum.argmin()
        return a_t

    def LWKR(s_t):
        data = s_t[:4,:]
        sum = data[:,1]
        a_t = sum.argmin()
        return a_t


'''
Neural network
'''
class network_value_based(nn.Module):
    def __init__(self, input_size, output_size):
        super(network_value_based, self).__init__()
        self.lr = 0.005
        self.input_size = input_size
        self.output_size = output_size
        self.flattened_input_size = torch.tensor(self.input_size).prod()
        # FCNN parameters
        layer_1 = 64
        layer_2 = 48
        layer_3 = 48
        layer_4 = 36
        layer_5 = 24
        layer_6 = 12
        # normalization modules
        self.norm_layer = nn.Sequential(
                                nn.LayerNorm(self.input_size),
                                nn.Flatten()
                                )
        # shared layers of machines
        self.FC_layers = nn.Sequential(
                                nn.Linear(self.flattened_input_size, layer_1),
                                nn.Tanh(),
                                nn.Linear(layer_1, layer_2),
                                nn.Tanh(),
                                nn.Linear(layer_2, layer_3),
                                nn.Tanh(),
                                nn.Linear(layer_3, layer_4),
                                nn.Tanh(),
                                nn.Linear(layer_4, layer_5),
                                nn.Tanh(),
                                nn.Linear(layer_5, layer_6),
                                nn.Tanh(),
                                nn.Linear(layer_6, output_size)
                                )
        # Huber loss function
        self.loss_func = F.smooth_l1_loss
        # the universal network for all scheudling agents
        self.network = nn.ModuleList([self.norm_layer, self.FC_layers])
        # accompanied by optimizer
        self.optimizer = optim.SGD(self.network.parameters(), lr=self.lr, momentum = 0.9)

    def forward(self, x, *args):
        #print('original',x)
        x = self.network[0](x)
        x = self.network[1](x)
        #print('output',x)
        return x

class network_TEST(nn.Module):
    def __init__(self, input_size, output_size):
        super(network_TEST, self).__init__()
        self.lr = 0.005
        self.input_size = input_size
        self.output_size = output_size
        self.flattened_input_size = torch.tensor(self.input_size).prod()
        # FCNN parameters
        layer_1 = 64
        layer_2 = 48
        layer_3 = 48
        layer_4 = 36
        layer_5 = 24
        layer_6 = 12
        # normalization modules
        self.norm_layer = nn.Sequential(
                                nn.LayerNorm(self.input_size),
                                nn.Flatten()
                                )
        # shared layers of machines
        self.FC_layers = nn.Sequential(
                                nn.Linear(self.flattened_input_size, layer_1),
                                nn.Tanh(),
                                nn.Linear(layer_1, layer_2),
                                nn.Tanh(),
                                nn.Linear(layer_2, layer_3),
                                nn.Tanh(),
                                nn.Linear(layer_3, layer_4),
                                nn.Tanh(),
                                nn.Linear(layer_4, layer_5),
                                nn.Tanh(),
                                nn.Linear(layer_5, layer_6),
                                nn.Tanh(),
                                nn.Linear(layer_6, output_size)
                                )
        # Huber loss function
        self.loss_func = F.smooth_l1_loss
        # the universal network for all scheudling agents
        self.network = nn.ModuleList([self.norm_layer, self.FC_layers])
        # accompanied by optimizer
        self.optimizer = optim.SGD(self.network.parameters(), lr=self.lr, momentum = 0.9)

    def forward(self, x, *args):
        #print('original',x)
        x = self.network[0](x)
        x = self.network[1](x)
        #print('output',x)
        return x

