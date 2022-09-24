import random
import numpy as np
import sys
import copy
import matplotlib.pyplot as plt

sys.path  # call this otherwise it will raise excepetion if you use another computer
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from tabulate import tabulate

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
        and specify the address to store the trained state-dict
        needs to be specified in kwargs, otherwise abstract networks + abstract state space
        NOTE there is an action_NN that perform the actual action and be trained
        and a target_NN to improve the stability of training
        '''
        if 'DDQN_SI' in kwargs and kwargs['DDQN_SI']: # strategic idleness is on
            print("---> STRATEGIC IDLENESS MODE (DDQN) mode ON <---")
            self.input_size =  self.state_direct(self.m_list[0].sequencing_data_generation()).size()
            self.input_size_as_list = list(self.input_size)
            self.output_size = 5
            self.action_NN = network_value_based(self.input_size, self.output_size)
            self.target_NN = copy.deepcopy(self.action_NN)
            self.address_seed = "{}\\trained_models\\DDQN_SI_rwd"+str(kwargs['reward_function'])+".pt"
            self.build_state = self.state_direct
            self.train = self.train_Double_DQN
            self.action_DRL = self.action_direct_SI
            for m in self.m_list:
                m.build_state = self.state_direct
        elif 'A2C' in kwargs and kwargs['A2C']: # advantage actor critic
            print("---> A2C MODE ON <---")
            self.input_size =  self.state_direct(self.m_list[0].sequencing_data_generation()).size()
            self.input_size_as_list = list(self.input_size)
            self.output_size = 4
            self.action_NN = network_ActorCritic(self.input_size, self.output_size)
            self.target_NN = copy.deepcopy(self.action_NN)
            self.address_seed = "{}\\trained_models\\A2C_rwd"+str(kwargs['reward_function'])+".pt"
            self.build_state = self.state_direct
            self.train = self.train_A2C
            self.action_DRL = self.action_A2C
            for m in self.m_list:
                m.build_state = self.state_direct
        elif 'IQL' in kwargs and kwargs['IQL']: # baseline, independent DQN agents
            print("---!!! BASELINE Independent DQN mode ON !!!---")
            self.input_size =  self.state_direct(self.m_list[0].sequencing_data_generation()).size()
            self.input_size_as_list = list(self.input_size)
            self.output_size = 4
            self.action_NN = network_independent(self.input_size, self.output_size, self.m_no)
            self.target_NN = copy.deepcopy(self.action_NN)
            self.address_seed = "{}\\trained_models\\independent\\IQL_rwd"+str(kwargs['reward_function'])+"_{}.pt"
            self.build_state = self.state_direct
            #self.train = self.train_prioritized_DDQN
            self.train = self.train_IQL
            self.action_DRL = self.action_direct
            for m in self.m_list:
                m.build_state = self.state_direct
        elif 'I_DDQN' in kwargs and kwargs['I_DDQN']: # baseline, independent DQN agents
            print("---!!! Independent Double DQN mode ON !!!---")
            self.input_size =  self.state_direct(self.m_list[0].sequencing_data_generation()).size()
            self.input_size_as_list = list(self.input_size)
            self.output_size = 4
            self.action_NN = network_independent(self.input_size, self.output_size, self.m_no)
            self.target_NN = copy.deepcopy(self.action_NN)
            self.address_seed = "{}\\trained_models\\independent\\I_DDQN_rwd"+str(kwargs['reward_function'])+"_{}.pt"
            self.build_state = self.state_direct
            #self.train = self.train_prioritized_DDQN
            self.train = self.train_I_DDQN
            self.action_DRL = self.action_direct
            for m in self.m_list:
                m.build_state = self.state_direct
        elif 'TEST' in kwargs and kwargs['TEST']:
            print("---!!! TEST mode ON !!!---")
            self.input_size =  self.state_direct(self.m_list[0].sequencing_data_generation()).size()
            self.input_size_as_list = list(self.input_size)
            self.output_size = 4
            self.action_NN = network_TEST(self.input_size, self.output_size)
            self.target_NN = copy.deepcopy(self.action_NN)
            self.address_seed = "{}\\trained_models\\TEST_DDQN_rwd"+str(kwargs['reward_function'])+".pt"
            self.build_state = self.state_direct
            #self.train = self.train_prioritized_DDQN
            self.train = self.train_DQN
            self.action_DRL = self.action_direct
            for m in self.m_list:
                m.build_state = self.state_direct
        elif 'N_2019' in kwargs and kwargs['N_2019']: # advantage actor critic
            print("---> benchmark N_2019 ON <---")
            self.input_size =  self.state_direct(self.m_list[0].sequencing_data_generation()).size()
            self.input_size_as_list = list(self.input_size)
            self.func_list = [sqc_func.SPT, sqc_func.LWKR, sqc_func.WINQ]
            self.output_size = 3
            self.action_NN = network_benchmark_value_based(self.input_size, self.output_size)
            self.target_NN = copy.deepcopy(self.action_NN)
            self.address_seed = "{}\\trained_models\\N_2019_rwd"+str(kwargs['reward_function'])+".pt"
            self.build_state = self.state_direct
            self.train = self.train_DQN
            self.action_DRL = self.action_sqc_rule_value_based
            for m in self.m_list:
                m.build_state = self.state_direct
        elif 'U_2020' in kwargs and kwargs['U_2020']:
            print("---> benchmark U_2020 ON <---")
            self.input_size =  self.state_direct(self.m_list[0].sequencing_data_generation()).size()
            self.input_size_as_list = list(self.input_size)
            self.func_list = [sqc_func.SPT, sqc_func.LWKR, sqc_func.WINQ]
            self.output_size = 3
            self.action_NN = network_benchmark_policy(self.input_size, self.output_size)
            self.target_NN = copy.deepcopy(self.action_NN)
            self.address_seed = "{}\\trained_models\\U_2020_rwd"+str(kwargs['reward_function'])+".pt"
            self.build_state = self.state_direct
            self.train = self.train_A2C
            self.action_DRL = self.action_sqc_rule_policy
            for m in self.m_list:
                m.build_state = self.state_direct
        elif 'O_2021' in kwargs and kwargs['O_2021']:
            print("---> benchmark O_2021 ON <---")
            self.input_size =  self.state_O_2021(self.m_list[0].sequencing_data_generation()).size()
            self.input_size_as_list = list(self.input_size)
            self.output_size = 4
            self.action_NN = network_value_based(self.input_size, self.output_size)
            self.target_NN = copy.deepcopy(self.action_NN)
            self.address_seed = "{}\\trained_models\\O_2021_rwd"+str(kwargs['reward_function'])+".pt"
            self.build_state = self.state_O_2021
            self.train = self.train_DQN
            self.action_DRL = self.action_direct
            for m in self.m_list:
                m.build_state = self.state_O_2021
        elif 'TEST_AS' in kwargs and kwargs['TEST_AS']:
            print("---!!! TEST + Abstracted state mode ON !!!---")
            self.input_size =  len(self.state_multi_channel(self.m_list[0].sequencing_data_generation()))
            self.input_size_as_list = [1,self.input_size]
            self.func_list = [Rule_sequencing.SPT, Rule_sequencing.WINQ, Rule_sequencing.MS, Rule_sequencing.CR]
            self.output_size = 4
            self.action_NN = network_TEST_AS(self.input_size, self.output_size)
            self.target_NN = copy.deepcopy(self.action_NN)
            self.address_seed = "{}\\trained_models\\TEST_AS_rwd"+str(kwargs['reward_function'])+".pt"
            self.build_state = self.state_multi_channel
            self.train = self.train_Double_DQN
            self.action_DRL = self.action_AS
            for m in self.m_list:
                m.build_state = self.state_multi_channel
        elif 'AS' in kwargs and kwargs['AS']:
            print("---!!! Abstracted state mode ON !!!---")
            self.input_size =  len(self.state_multi_channel(self.m_list[0].sequencing_data_generation()))
            self.input_size_as_list = [1,self.input_size]
            self.func_list = [Rule_sequencing.SPT, Rule_sequencing.WINQ, Rule_sequencing.MS, Rule_sequencing.CR]
            self.output_size = 4
            self.action_NN = network_AS(self.input_size, self.output_size)
            self.target_NN = copy.deepcopy(self.action_NN)
            self.address_seed = "{}\\trained_models\\Abstracted_state_rwd"+str(kwargs['reward_function'])+".pt"
            self.build_state = self.state_multi_channel
            self.train = self.train_Double_DQN
            self.action_DRL = self.action_AS
            for m in self.m_list:
                m.build_state = self.state_multi_channel
        elif 'IQL_AS' in kwargs and kwargs['IQL_AS']: # baseline, independent DQN agents
            print("---!!! BASELINE Independent Abstracted DQN mode ON !!!---")
            self.input_size =  len(self.state_multi_channel(self.m_list[0].sequencing_data_generation()))
            self.input_size_as_list = [1,self.input_size]
            self.func_list = [Rule_sequencing.SPT, Rule_sequencing.WINQ, Rule_sequencing.MS, Rule_sequencing.CR]
            self.output_size = 4
            self.action_NN = network_independent_AS(self.input_size, self.output_size, self.m_no)
            self.target_NN = copy.deepcopy(self.action_NN)
            self.address_seed = "{}\\trained_models\\independent\\IQL_AS_rwd"+str(kwargs['reward_function'])+"_{}.pt"
            self.build_state = self.state_multi_channel
            #self.train = self.train_prioritized_DDQN
            self.train = self.train_IQL
            self.action_DRL = self.action_AS
            for m in self.m_list:
                m.build_state = self.state_multi_channel
        elif 'I_DDQN_AS' in kwargs and kwargs['I_DDQN_AS']: # baseline, independent DQN agents
            print("---!!! BASELINE Independent Abstracted double DQN mode ON !!!---")
            self.input_size =  len(self.state_multi_channel(self.m_list[0].sequencing_data_generation()))
            self.input_size_as_list = [1,self.input_size]
            self.func_list = [Rule_sequencing.SPT, Rule_sequencing.WINQ, Rule_sequencing.MS, Rule_sequencing.CR]
            self.output_size = 4
            self.action_NN = network_independent_AS(self.input_size, self.output_size, self.m_no)
            self.target_NN = copy.deepcopy(self.action_NN)
            self.address_seed = "{}\\trained_models\\independent\\I_DDQN_AS_rwd"+str(kwargs['reward_function'])+"_{}.pt"
            self.build_state = self.state_multi_channel
            #self.train = self.train_prioritized_DDQN
            self.train = self.train_I_DDQN
            self.action_DRL = self.action_AS
            for m in self.m_list:
                m.build_state = self.state_multi_channel
        else: # the default mode is DDQN
            print("---X DEFAULT (DDQN) mode ON X---")
            self.input_size =  self.state_direct(self.m_list[0].sequencing_data_generation()).size()
            self.input_size_as_list = list(self.input_size)
            self.output_size = 4
            self.action_NN = network_value_based(self.input_size, self.output_size)
            self.target_NN = copy.deepcopy(self.action_NN)
            self.address_seed = "{}\\trained_models\\DDQN_rwd"+str(kwargs['reward_function'])+".pt"
            self.build_state = self.state_direct
            self.train = self.train_Double_DQN
            self.action_DRL = self.action_direct
            for m in self.m_list:
                m.build_state = self.state_direct
        '''
        sometimes train based on trained parameters can save a lot of time
        importing trained parameters from specified address
        '''
        if kwargs['bsf_start']: # import best for far trained parameters to kick off
            if kwargs['TEST']:
                import_address = "{}\\trained_models\\bsf_TEST.pt"
            elif kwargs['A2C']:
                import_address = "{}\\trained_models\\bsf_A2C.pt"
            else:
                import_address = "{}\\trained_models\\bsf_DDQN.pt"
            self.action_NN.network.load_state_dict(torch.load(import_address.format(sys.path[0])))
            print("IMPORT FROM:", import_address)
        '''
        new address seed for storing the trained parameters, if specified
        '''
        if 'store_to' in kwargs:
            self.address_seed = "{}\\trained_models\\" + str(kwargs['store_to']) + ".pt"
            print("New address seed:", self.address_seed)
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
        '''
        exploration mode
        '''
        if kwargs['DDQN_SI']:
            self.env.process(self.warm_up_process())
            for m in self.m_list:
                m.job_sequencing = self.random_exploration_SI
        elif kwargs['expert']:
            self.env.process(self.expert_warm_up_process())
            for m in self.m_list:
                m.job_sequencing = self.expert_exploration
        elif kwargs['AS'] or kwargs['TEST_AS'] or kwargs['IQL_AS'] or kwargs['I_DDQN_AS']:
            self.env.process(self.warm_up_process())
            for m in self.m_list:
                m.job_sequencing = self.random_exploration_AS
        else:
            self.env.process(self.warm_up_process())
            for m in self.m_list:
                m.job_sequencing = self.random_exploration
        '''
        training scheme, independent learning requires unique rep_memo and training
        '''
        if kwargs['IQL'] or kwargs['I_DDQN'] or kwargs['IQL_AS'] or kwargs['I_DDQN_AS']:
            self.env.process(self.training_process_independent())
            self.env.process(self.update_rep_memo_independent_process())
            self.rep_memo = {} # replace the list by dict
            for m in self.m_list:
                self.rep_memo[m.m_idx] = []
            self.build_initial_rep_memo = self.build_initial_rep_memo_independent
            #self.rep_memo_size /= self.m_no # size for independent replay memory
        else: # default mode is parameter sharing
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

    def expert_warm_up_process(self): # Warm-up phase with experience created by "experts"
        self.sqc_func = [sqc_func.DPTLWKRS, sqc_func.MDD, sqc_func.PTWINQS]
        # second half of warm-up period
        print("expert tutorial from time {} to time {}".format(self.env.now, self.warm_up))
        for func in self.sqc_func:
            print('use expert function %s at time %s'%(func.__name__, self.env.now))
            self.expert = func
            yield self.env.timeout((self.warm_up-1)/len(self.sqc_func))
        # after the warm up period, build replay memory and start training
        self.build_initial_rep_memo()
        # hand over the target machines' sequencing function to DRL (action network)
        for m in self.m_list:
            m.job_sequencing = self.action_DRL


    '''action functions for random exploration'''
    def random_exploration_SI(self, sqc_data):
        s_t = self.build_state(sqc_data)
        #print('state:',self.env.now,s_t)
        # action is a random index of job in action space
        a_t = torch.randint(self.output_size,[])
        '''check the feasibilioty of action'''
        # if choose currently queuing jobs
        if a_t != self.output_size - 1:
            self.strategic_idleness_bool = False # then no need do strategic idleness
        # if choose wait for arriving job and there is an arriving job
        elif a_t == self.output_size - 1 and self.arriving_job_exists:
            self.strategic_idleness_bool = True # implement strategic idleness
        # if choose wait for arriving job but there is NO arriving job
        else:
            a_t = torch.randint(self.output_size-1,[]) # roll the dice again
            self.strategic_idleness_bool = False # no strategic idleness
        job_position, j_idx = self.action_conversion(a_t)
        m_idx = sqc_data[-1]
        # add the state at sequencing to job creator's corresponding repository
        self.build_experience(j_idx,m_idx,s_t,a_t)
        return job_position, self.strategic_idleness_bool, self.strategic_idleness_time

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

    def expert_exploration(self, sqc_data):
        s_t = self.build_state(sqc_data)
        if random.random() < 0.3:
            a_t = torch.randint(0,self.output_size,[])
        else:
            a_t = self.expert(s_t)
        job_position, j_idx = self.action_conversion(a_t)
        self.strategic_idleness_bool = False # then no need do strategic idleness
        m_idx = sqc_data[-1]
        # add the state at sequencing to job creator's corresponding repository
        self.build_experience(j_idx,m_idx,s_t,a_t)
        return job_position, self.strategic_idleness_bool, self.strategic_idleness_time

    def random_exploration_AS(self, sqc_data):
        s_t = self.build_state(sqc_data)
        #print('state:',self.env.now,s_t)
        # action is a random index of job in action space
        a_t = torch.randint(self.output_size,[])
        # the decision is made by one of the sequencing rules
        job_position, self.strategic_idleness_bool, self.strategic_idleness_time = self.func_list[a_t](sqc_data)
        j_idx = sqc_data[-2][job_position]
        m_idx = sqc_data[-1]
        # add the state at sequencing to job creator's corresponding repository
        self.build_experience(j_idx,m_idx,s_t,a_t)
        return job_position, self.strategic_idleness_bool, self.strategic_idleness_time

    '''action functions for actual DRL control'''
    def action_direct_SI(self, sqc_data): # direct selection that allows strategic idleness
        s_t = self.build_state(sqc_data)
        m_idx = sqc_data[-1]
        # Epsilon-greedy policy
        if random.random() < self.epsilon:
            a_t = torch.randint(0,self.output_size,[])
            if a_t != self.output_size - 1:
                self.strategic_idleness_bool = False # then no need do strategic idleness
            # if choose wait for arriving job and there is an arriving job
            if a_t == self.output_size - 1 and self.arriving_job_exists:
                self.strategic_idleness_bool = True # implement strategic idleness
            # if choose wait for arriving job but there is NO arriving job
            else:
                a_t = torch.randint(0,self.output_size-1,[]) # eliminate the infeasible action
                self.strategic_idleness_bool = False # no strategic idleness
            print('Random Selection:', a_t)
        else:
            # input state to action network, produce the state-action value
            value = self.action_NN.forward(s_t.reshape([1]+self.input_size_as_list),m_idx).squeeze()
            # greedy policy
            a_t = value.argmax()
            #print("State is:", s_t)
            #print('State-Action Values:', value)
            if a_t != self.output_size - 1:
                self.strategic_idleness_bool = False # then no need do strategic idleness
            if a_t == self.output_size - 1 and self.arriving_job_exists:
                self.strategic_idleness_bool = True # implement strategic idleness
            else:
                a_t = value[:self.output_size-1].argmax() # eliminate the infeasible action
                self.strategic_idleness_bool = False # no strategic idleness
            print('DRL Selection: %s'%(a_t))
        job_position, j_idx = self.action_conversion(a_t)
        # add the state at sequencing to job creator's corresponding repository
        self.build_experience(j_idx,m_idx,s_t,a_t)
        return job_position, self.strategic_idleness_bool, self.strategic_idleness_time

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

    def action_A2C(self, sqc_data):
        s_t = self.build_state(sqc_data)
        # input state to actor network, produce the probability
        prob = self.action_NN.actor_forward(s_t.reshape([1]+self.input_size_as_list)).squeeze()
        # stochastic policy
        a_t = Categorical(prob).sample()
        #print("State is:", s_t)
        #print('prob:', prob)
        #print('A2C Action:%s'%(a_t))
        self.strategic_idleness_bool = False # then no need do strategic idleness
        job_position, j_idx = self.action_conversion(a_t)
        m_idx = sqc_data[-1]
        # add the state at sequencing to job creator's corresponding repository
        self.build_experience(j_idx,m_idx,s_t,a_t)
        return job_position, self.strategic_idleness_bool, self.strategic_idleness_time

    def action_sqc_rule_value_based(self, sqc_data):
        s_t = self.build_state(sqc_data)
        m_idx = sqc_data[-1]
        if random.random() < self.epsilon:
            a_t = torch.randint(0,self.output_size,[])
        else:
            value = self.action_NN.forward(s_t.reshape([1]+self.input_size_as_list),m_idx).squeeze()
            a_t = torch.argmax(value)
        self.strategic_idleness_bool = False
        rule_a_t = self.func_list[a_t](s_t)
        job_position, j_idx = self.action_conversion(rule_a_t)
        self.build_experience(j_idx,m_idx,s_t,a_t)
        return job_position, self.strategic_idleness_bool, self.strategic_idleness_time

    def action_sqc_rule_policy(self, sqc_data):
        s_t = self.build_state(sqc_data)
        m_idx = sqc_data[-1]
        prob = self.action_NN.actor_forward(s_t.reshape([1]+self.input_size_as_list)).squeeze()
        a_t = Categorical(prob).sample()
        self.strategic_idleness_bool = False
        rule_a_t = self.func_list[a_t](s_t)
        job_position, j_idx = self.action_conversion(rule_a_t)
        self.build_experience(j_idx,m_idx,s_t,a_t)
        return job_position, self.strategic_idleness_bool, self.strategic_idleness_time

    def action_AS(self, sqc_data):
        s_t = self.build_state(sqc_data)
        m_idx = sqc_data[-1]
        # Epsilon-greedy policy
        if random.random() < self.epsilon:
            a_t = torch.randint(0,self.output_size,[])
            #print('Random Action / By Brain')
        else:
            # input state to policy network, produce the state-action value
            value = self.action_NN.forward(s_t.reshape([1,1,self.input_size]),m_idx)
            # greedy policy
            a_t = torch.argmax(value)
            #print("State is:", s_t)
            #print('State-Action Values:', value)
            #print('Sequencing NN, action %s / By Brain'%(a_t))
        job_position, self.strategic_idleness_bool, self.strategic_idleness_time = self.func_list[a_t](sqc_data)
        j_idx = sqc_data[-2][job_position]
        # add the state at sequencing to job creator's corresponding repository
        self.build_experience(j_idx,m_idx,s_t,a_t)
        return job_position, self.strategic_idleness_bool, self.strategic_idleness_time

    '''
    2. downwards are functions used for building the state of the experience (replay memory)
       those functions are also used by validation module

    data consists of:
    0            1                 2         3        4              5      6     7
    [current_pt, remaining_job_pt, due_list, env.now, time_till_due, slack, winq, avlm,
    8        9                10           11/-2  12/-1
    next_pt, remaining_no_op, waited_time, queue, m_idx]

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

    def state_multi_channel(self, sqc_data):
        # information in job number, global and local
        in_system_job_no = self.job_creator.in_system_job_no
        local_job_no = len(sqc_data[0])
        # the information of arriving job (currently being processed by other machines)
        #print('arriving jobs:',self.job_creator.next_wc_list, self.job_creator.arriving_job_slack_list, self.job_creator.next_output_list, sqc_data[-4])
        arriving_jobs = np.where(self.job_creator.next_machine_list == sqc_data[-1])[0] # see if there are jobs will arrive
        arriving_job_no = arriving_jobs.size  # expected arriving job number
        if arriving_job_no: # if there're jobs arriving at your machine
            # get the exact next job that will arrive at machine out of all arriving jobs
            pos = arriving_jobs[self.job_creator.release_time_list[arriving_jobs].argmin()]
            arriving_j_idx = self.job_creator.current_j_idx_list[pos]
            arriving_job_time = self.job_creator.release_time_list[arriving_jobs].min() - self.env.now
            arriving_job_slack = self.job_creator.arriving_job_slack_list[pos]
        else:
            arriving_job_time = 0
            arriving_job_slack = 0
        #print(arriving_job_idx, arriving_job_no, arriving_job_time, arriving_job_slack, self.env.now, sqc_data[-4])
        # information of progression of jobs, get from the job creator
        global_comp_rate = self.job_creator.comp_rate
        global_realized_tard_rate = self.job_creator.realized_tard_rate
        global_exp_tard_rate = self.job_creator.exp_tard_rate
        available_time = (self.job_creator.available_time_list - self.env.now).clip(0,None)
        # get the pt of all remaining jobs in system
        rem_pt = []
        # need loop here because remaining_pt have different length
        for m in self.m_list:
            for x in m.remaining_pt_list:
                rem_pt += x.tolist()
        # processing time related data
        pt_share = available_time[sqc_data[-1]] / sum(available_time) # sum of pt / sum of available time
        global_pt_CV = np.std(rem_pt) / np.mean(rem_pt)
        # information of queuing jobs in queue
        local_pt_sum = np.sum(sqc_data[0])
        local_pt_mean = np.mean(sqc_data[0])
        local_pt_min = np.min(sqc_data[0])
        local_pt_CV = np.std(sqc_data[0]) / local_pt_mean
        # information of queuing jobs in remaining processing time
        local_remaining_pt_sum = np.sum(sqc_data[1])
        local_remaining_pt_mean = np.mean(sqc_data[1])
        local_remaining_pt_max = np.max(sqc_data[1])
        local_remaining_pt_CV = np.std(sqc_data[1]) / local_remaining_pt_mean
        # information of WINQ
        avlm_mean = np.mean(sqc_data[7])
        avlm_min = np.min(sqc_data[7])
        avlm_CV = np.std(sqc_data[7]) / avlm_mean
        # time-till-due related data:
        time_till_due = sqc_data[4]
        realized_tard_rate = time_till_due[time_till_due<0].size / local_job_no # ratio of tardy jobs
        ttd_sum = time_till_due.sum()
        ttd_mean = time_till_due.mean()
        ttd_min = time_till_due.min()
        ttd_CV = (time_till_due.std() / ttd_mean).clip(-2,2)
        # slack-related data:
        slack = sqc_data[5]
        exp_tard_rate = slack[slack<0].size / local_job_no # ratio of jobs expect to be tardy
        slack_sum = slack.sum()
        slack_mean = slack.mean()
        slack_min = slack.min()
        slack_CV = (slack.std() / slack_mean).clip(-2,2)
        # use raw data, and leave the magnitude adjustment to normalization layers
        no_info = [in_system_job_no, arriving_job_no, local_job_no] # info in job number
        pt_info = [local_pt_sum, local_pt_mean, local_pt_min] # info in processing time
        remaining_pt_info = [local_remaining_pt_sum, local_remaining_pt_mean, local_remaining_pt_max, avlm_mean, avlm_min] # info in remaining processing time
        ttd_slack_info = [ttd_mean, ttd_min, slack_mean, slack_min, arriving_job_slack] # info in time till due
        progression = [pt_share, global_comp_rate, global_realized_tard_rate, global_exp_tard_rate] # progression info
        heterogeneity = [global_pt_CV, local_pt_CV, ttd_CV, slack_CV, avlm_CV] # heterogeneity
        # concatenate the data input
        s_t = np.nan_to_num(np.concatenate([no_info, pt_info, remaining_pt_info, ttd_slack_info, progression, heterogeneity]),nan=0,posinf=1,neginf=-1)
        # convert to tensor
        s_t = torch.tensor(s_t, dtype=torch.float)
        return s_t

    def state_O_2021(self, sqc_data):
        '''STEP 1: check queuing jobs, if any, clip the sqc data'''
        # number of candidate jobs
        no_candidate_jobs = len(sqc_data[0])
        if no_candidate_jobs == 1: # if there's only one queuing job, simply copy the info of the only job (most common case)
            # original sqc_data contains lots of things that won't be used, create a clipped copy of it
            clipped_data = np.concatenate([sqc_data[0], sqc_data[1], sqc_data[5], sqc_data[7], sqc_data[10]])
            s_t = [clipped_data[:2] for i in range(4)]
            # and set the correspondence to the first job in the queue
            self.correspondence_pos = [0 for i in range(4)]
            self.correspondence_idx = [0 for i in range(4)]
        elif no_candidate_jobs == 0 : # if there's no queuing job, create dummy state that all enrties are 0
            s_t = [np.array([0 for i in range(2)]) for i in range(4)]
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
                    s_t.append(copied_clipped_data[:2,no_duplication_pos])
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
                    s_t.append(clipped_data[:2,normal_pos])
                    row += 1
            self.strategic_idleness_time = 0 # no need to wait for any arriving jobs
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

    def build_initial_rep_memo_independent(self):
        #print(self.job_creator.rep_memo)
        print('INITIALIZATION - replay_memory')
        for m in self.m_list:
            # copy the initial memoery from corresponding rep_memo from job creator
            #print('%s complete and %s incomplete experience for machine %s'%(len(self.job_creator.rep_memo[m.m_idx]), len(self.job_creator.incomplete_rep_memo[m.m_idx]), m.m_idx))
            #print(self.job_creator.incomplete_rep_memo[m.m_idx])
            self.rep_memo[m.m_idx] += self.job_creator.rep_memo[m.m_idx].copy()
            # and clear the replay memory in job creator, keep it updated
            self.job_creator.rep_memo[m.m_idx] = []
            print(tabulate(self.rep_memo[m.m_idx], headers = ['s_t','a_t','s_t+1','r_t']))
            print('INITIALIZATION - size of replay memory:',len(self.rep_memo[m.m_idx]))
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

    def update_rep_memo_independent_process(self):
        yield self.env.timeout(self.warm_up)
        while self.env.now < self.span:
            for m in self.m_list:
                # add new memoery from corresponding rep_memo from job creator
                self.rep_memo[m.m_idx] += self.job_creator.rep_memo[m.m_idx].copy()
                # and assign top priority to new experiences
                #self.rep_memo_TDerror = torch.cat([self.rep_memo_TDerror, torch.ones(len(self.job_creator.rep_memo[m.m_idx]),dtype=torch.float)])
                # and clear the replay memory in job creator, keep it updated
                self.job_creator.rep_memo[m.m_idx] = []
            # clear the obsolete experience periodically
            if len(self.rep_memo[m.m_idx]) > self.rep_memo_size:
                truncation = len(self.rep_memo[m.m_idx])-self.rep_memo_size
                self.rep_memo[m.m_idx] = self.rep_memo[m.m_idx][truncation:]
                #self.rep_memo_TDerror = self.rep_memo_TDerror[truncation:]
            #print(self.rep_memo_TDerror)
            yield self.env.timeout(self.action_NN_training_interval*10)

    '''
    4. downwards are functions used in the training of DRL, including the dynamic training process
       dynamic training parameters update
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
        # specify the address to store the model / state_dict
        address = self.address_seed.format(sys.path[0])
        # save the parameters of policy / action network after training
        torch.save(self.action_NN.network.state_dict(), address)
        # after the training, print out the setting of DRL architecture
        print("Training terminated, store trained parameters to: {}".format(self.address_seed))

    # agents don't share parameters of network
    def training_process_independent(self):
        yield self.env.timeout(self.warm_up)
        for i in range(10):
            for m in self.m_list:
                self.train(m.m_idx)
        while self.env.now < self.span:
            for m in self.m_list:
                self.train(m.m_idx)
            yield self.env.timeout(self.action_NN_training_interval)
        for m in self.m_list:
            print('FINAL - replay_memory of machine %s is:'%m.m_idx)
            print(tabulate(self.rep_memo[m.m_idx],headers = ['s_t','a_t','s_t+1','r_t']))
            address = self.address_seed.format(sys.path[0],str(m.m_idx))
            torch.save(self.action_NN.network_dict[m.m_idx].state_dict(), address)
        print("Training terminated, store trained parameters to: {}".format(self.address_seed))

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

    def train_A2C(self):
        size = min(len(self.rep_memo),self.minibatch_size)
        minibatch = random.sample(self.rep_memo,size)
        '''
        slice, and stack the 1D tensors to several 3D tensors (batch, channel, vector)
        the "torch.stack" is only applicable when the augment is a list of tensors, or multi-dimensional tensor
        '''
        #print(minibatch)
        sample_s0_batch = torch.stack([data[0] for data in minibatch], dim=0).reshape([size]+self.input_size_as_list)
        sample_s1_batch = torch.stack([data[2] for data in minibatch], dim=0).reshape([size]+self.input_size_as_list)
        sample_a0_batch = torch.stack([data[1] for data in minibatch], dim=0)
        sample_r0_batch = torch.stack([data[3] for data in minibatch], dim=0).reshape(size,1)
        '''
        first get the TD error to train the critic (value)
        TD error is also the advantage that needed for training actor (policy)
        '''
        # get the state value of s0 and s1, by critic network
        V_0 = self.action_NN.critic_forward(sample_s0_batch)
        #print('V_0 is:\n', V_0)
        V_1 = self.target_NN.critic_forward(sample_s1_batch).detach()
        #print('V_1 is:\n', V_1)
        target_value = sample_r0_batch + self.discount_factor * V_1
        #print('critic target value/ return is:\n', target_value)
        Advantage = (target_value - V_0.detach()).squeeze() # also the TD error
        #print('advantage/ TD error is:\n', Advantage)
        critic_loss = self.action_NN.critic_loss_func(V_0, target_value)
        #print('critic loss:',critic_loss)
        '''
        use the logrithmed probability, NOT probability (refer to policy gradient theorem)
        change the sign of "loss", to perform gradient ascent, NOT descent!
        '''
        # get the probability of all actions, by actor network
        prob_all = self.action_NN.actor_forward(sample_s0_batch)
        #print("probability is:", prob_all)
        #print("action is", sample_a0_batch)
        dist_all = Categorical(prob_all) # returns a distribution
        log_prob_action = dist_all.log_prob(sample_a0_batch) # get the logrithmed value of distribution
        #print("log prob of actions:", log_prob_action)
        actor_loss = - (log_prob_action * Advantage).mean() # add NEGATIVE sign to do gradient ASCENT !!!
        #print('actor loss:',actor_loss)
        # printout
        if not self.env.now % 50:
            print('Time: %s, actor loss: %s, critic loss: %s'%(self.env.now, actor_loss, critic_loss))
        self.loss_time_record.append(self.env.now)
        self.loss_record.append(float(actor_loss))
        # clear the old gradient of parameters for both actor and critic
        self.action_NN.actor_optimizer.zero_grad()
        self.action_NN.critic_optimizer.zero_grad()
        # get the new gradient of parameters
        actor_loss.backward(retain_graph=True)
        critic_loss.backward(retain_graph=True)
        '''
        # check the gradient, to avoid exploding/vanishing gradient, very seldom though
        for param in self.action_NN.module_dict[m_idx].parameters():
            print(param.grad.norm())
        '''
        # update the parameters
        self.action_NN.actor_optimizer.step()
        self.action_NN.critic_optimizer.step()

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

    def train_prioritized_DDQN(self):
        """
        prioritized experience replay
        every element in the replay menory is a list [s_0, a_0, s_1, r_0]
        all element of this list are tensors
        """
        size = self.minibatch_size
        minibatch_idx = Categorical(self.rep_memo_TDerror).sample([self.minibatch_size])
        #print(len(self.rep_memo),self.rep_memo_TDerror.size(),minibatch_idx)
        # stack the data
        sample_s0_batch = torch.stack([self.rep_memo[idx][0] for idx in minibatch_idx])
        sample_a0_batch = torch.stack([self.rep_memo[idx][1] for idx in minibatch_idx]).reshape(size,1)
        sample_s1_batch = torch.stack([self.rep_memo[idx][2] for idx in minibatch_idx])
        sample_r0_batch = torch.stack([self.rep_memo[idx][3] for idx in minibatch_idx]).reshape(size,1)
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
        abs_TDerror = (target_value - current_value).abs() # get absolute TD error
        #print(target_value, current_value, abs_TDerror)
        self.rep_memo_TDerror[minibatch_idx] = torch.clamp(abs_TDerror.squeeze(),0,1)
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

    def train_DQN(self):
        size = min(len(self.rep_memo),self.minibatch_size)
        minibatch = random.sample(self.rep_memo,size)
        sample_s0_batch = torch.stack([data[0] for data in minibatch], dim=0).reshape([size]+self.input_size_as_list)
        sample_s1_batch = torch.stack([data[2] for data in minibatch], dim=0).reshape([size]+self.input_size_as_list)
        sample_a0_batch = torch.stack([data[1] for data in minibatch], dim=0).reshape(size,1)
        sample_r0_batch = torch.stack([data[3] for data in minibatch], dim=0).reshape(size,1)
        Q_0 = self.action_NN.forward(sample_s0_batch)
        current_value = Q_0.gather(1, sample_a0_batch)
        Q_1 = self.target_NN.forward(sample_s1_batch).detach()
        max_Q_1, max_Q_1_idx = torch.max(Q_1, dim=1)
        next_state_value = (self.discount_factor * max_Q_1).reshape([size,1])
        target_value = (sample_r0_batch + next_state_value)
        loss = self.action_NN.loss_func(current_value, target_value)
        self.loss_time_record.append(self.env.now)
        self.loss_record.append(float(loss))
        if not self.env.now%50:
            print('Time: %s, loss: %s:'%(self.env.now, loss))
        self.action_NN.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.action_NN.optimizer.step()

    def train_IQL(self, m_idx): # VANILLA independent DQN, as the baseline
        size = min(len(self.rep_memo[m_idx]),self.minibatch_size)
        minibatch = random.sample(self.rep_memo[m_idx],size)
        sample_s0_batch = torch.stack([data[0] for data in minibatch], dim=0).reshape([size]+self.input_size_as_list)
        sample_s1_batch = torch.stack([data[2] for data in minibatch], dim=0).reshape([size]+self.input_size_as_list)
        sample_a0_batch = torch.stack([data[1] for data in minibatch], dim=0).reshape(size,1)
        sample_r0_batch = torch.stack([data[3] for data in minibatch], dim=0).reshape(size,1)
        Q_0 = self.action_NN.forward(sample_s0_batch, m_idx)
        current_value = Q_0.gather(1, sample_a0_batch)
        Q_1 = self.target_NN.forward(sample_s1_batch, m_idx).detach()
        max_Q_1, max_Q_1_idx = torch.max(Q_1, dim=1) # use action network to get action, rather than max operation
        next_state_value = (self.discount_factor * max_Q_1).reshape([size,1])
        target_value = (sample_r0_batch + next_state_value)
        loss = self.action_NN.loss_func(current_value, target_value)
        self.loss_time_record.append(self.env.now)
        self.loss_record.append(float(loss))
        self.action_NN.optimizer_dict[m_idx].zero_grad()
        loss.backward(retain_graph=True)
        self.action_NN.optimizer_dict[m_idx].step()

    def train_I_DDQN(self, m_idx): # VANILLA independent DQN, as the baseline
        size = min(len(self.rep_memo[m_idx]),self.minibatch_size)
        minibatch = random.sample(self.rep_memo[m_idx],size)
        sample_s0_batch = torch.stack([data[0] for data in minibatch], dim=0).reshape([size]+self.input_size_as_list)
        sample_s1_batch = torch.stack([data[2] for data in minibatch], dim=0).reshape([size]+self.input_size_as_list)
        sample_a0_batch = torch.stack([data[1] for data in minibatch], dim=0).reshape(size,1)
        sample_r0_batch = torch.stack([data[3] for data in minibatch], dim=0).reshape(size,1)
        Q_0 = self.action_NN.forward(sample_s0_batch, m_idx)
        current_value = Q_0.gather(1, sample_a0_batch)
        Q_1_action = self.action_NN.forward(sample_s1_batch, m_idx).detach()
        Q_1_target = self.target_NN.forward(sample_s1_batch, m_idx).detach()
        max_Q_1_action, max_Q_1_action_idx = torch.max(Q_1_action, dim=1) # use action network to get action, rather than max operation
        max_Q_1_action_idx = max_Q_1_action_idx.reshape([size,1])
        next_state_value = Q_1_target.gather(1, max_Q_1_action_idx)
        next_state_value *= self.discount_factor
        target_value = (sample_r0_batch + next_state_value)
        loss = self.action_NN.loss_func(current_value, target_value)
        self.loss_time_record.append(self.env.now)
        self.loss_record.append(float(loss))
        self.action_NN.optimizer_dict[m_idx].zero_grad()
        loss.backward(retain_graph=True)
        self.action_NN.optimizer_dict[m_idx].step()

    # print out the functions and classes used in the training
    def check_parameter(self):
        print('-------------  Training Setting Check  -------------')
        print("Address seed:",self.address_seed)
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

    def loss_record_output(self,**kwargs):
        fig = plt.figure(figsize=(10,5.5))
        # left half, showing the loss of training
        axe = fig.add_subplot(1,1,1)
        axe.set_xlabel('Iterations of training ('+r'$\times 10^3$'+')')
        axe.set_ylabel('Loss (error) of training')
        iterations = np.arange(len(self.loss_record))
        print(self.loss_record, iterations)
        axe.scatter(iterations, self.loss_record,s=0.6,color='r', alpha=0.2)
        # moving average
        x = 50
        axe.plot(np.arange(x/2,len(self.loss_record)-x/2+1,1),np.convolve(self.loss_record, np.ones(x)/x, mode='valid'),color='k',label='moving average',zorder=3)
        # limits, grids,
        ylim_upper = 0.2
        ylim_lower = 0
        axe.set_xlim(0,len(self.loss_record))
        axe.set_ylim(ylim_lower,ylim_upper)
        xtick_interval = 1000
        axe.set_xticks(np.arange(0,len(self.loss_record)+1,xtick_interval))
        axe.set_xticklabels(np.arange(0,len(self.loss_record)/xtick_interval,1).astype(int),rotation=30, ha='right', rotation_mode="anchor", fontsize=8.5)
        axe.set_yticks(np.arange(ylim_lower, ylim_upper+0.01, 0.01))
        axe.grid(axis='x', which='major', alpha=0.5, zorder=0, )
        axe.grid(axis='y', which='major', alpha=0.5, zorder=0, )
        axe.legend()
        # dual axis
        ax_time = axe.twiny()
        ax_time.set_xlabel('Time in simulation ('+r'$\times 10^3$'+', excluding warm up phase)')
        ax_time.set_xlim(self.warm_up,self.span)
        ax_time.set_xticks(np.arange(self.warm_up,self.span+1,xtick_interval*2))
        ax_time.set_xticklabels(np.arange(self.warm_up/xtick_interval,self.span/xtick_interval+1,2).astype(int),rotation=30, ha='left', rotation_mode="anchor", fontsize=8.5)
        fig.subplots_adjust(top=0.8, bottom=0.1, right=0.9)
        plt.show()
        # save the loss record to .txt file
        record = open("trained_models\\training_loss_record.txt","w")
        record.write(str(self.loss_record))
        record.close()
        # save the figure if required
        if 'save' in kwargs and kwargs['save']:
            address = sys.path[0]+"\\experiment_result_figure\\training_loss_{}machines.png".format(len(self.m_list))
            fig.savefig(address, dpi=500, bbox_inches='tight')
            print('figure saved to'+address)
        return

    def reward_record_output(self,**kwargs):
        fig = plt.figure(figsize=(10,5.5))
        axe = fig.add_subplot(1,1,1)
        axe.set_xlabel('Time')
        axe.set_ylabel('Reward')
        data = self.job_creator.reward_record[0]
        time = np.array(data)[0]
        rewards = np.array(data)[1]
        print(self.job_creator.reward_record)
        print(time, rewards)
        axe.scatter(time, rewards, s=1,color='g', alpha=0.3, zorder=3)
        axe.set_xlim(0,self.span)
        axe.set_ylim(-1.1,0.1)
        xtick_interval = 2000
        axe.set_xticks(np.arange(0,self.span+1,xtick_interval))
        axe.set_xticklabels(np.arange(0,self.span+1,xtick_interval),rotation=30, ha='right', rotation_mode="anchor", fontsize=8.5)
        axe.set_yticks(np.arange(-1, 0, 0.1))
        axe.grid(axis='x', which='major', alpha=0.5, zorder=0, )
        axe.grid(axis='y', which='major', alpha=0.5, zorder=0, )
        # moving average
        x = 50
        print(len(time))
        axe.plot(time[int(x/2):len(time)-int(x/2)+1],np.convolve(rewards, np.ones(x)/x, mode='valid'),color='k',label="moving average")
        axe.legend()
        plt.show()
        # save the figure if required
        fig.subplots_adjust(top=0.5, bottom=0.5, right=0.9)
        if 'save' in kwargs and kwargs['save']:
            fig.savefig(sys.path[0]+"//experiment_result_figure//reward_{}m.png".format(len(self.m_list)), dpi=500, bbox_inches='tight')
        return

'''
class of sequencing functions to kick-off the training

0            1                 2         3        4              5      6     7
[current_pt, remaining_job_pt, due_list, env.now, time_till_due, slack, winq, avlm,
8        9                10           11/-2  12/-1
next_pt, remaining_no_op, waited_time, queue, m_idx]

clipped_data = np.concatenate([sqc_data[0], sqc_data[1], sqc_data[5], sqc_data[7], sqc_data[10]])
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
policy-based and actor-critic
'''
class network_ActorCritic(nn.Module):
    def __init__(self, input_size, output_size, **kwargs):
        super(network_ActorCritic, self).__init__()
        print("*** Initialize ActorCritic networks ***")
        self.input_size = input_size
        self.output_size = output_size
        self.lr = 0.001
        self.flattened_input_size = torch.tensor(self.input_size).prod()
        # FCNN parameters
        layer_1 = 48
        layer_2 = 36
        layer_3 = 36
        layer_4 = 24
        layer_5 = 24
        layer_6 = 12
        # shared fully-connected layers
        self.shared_FC_layers = nn.Sequential(
                                nn.LayerNorm(self.input_size),
                                nn.Flatten(),
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
                                )
        # output layer for actor and critic
        self.actor_output_layers = nn.Sequential(
                                nn.Linear(layer_6, self.output_size),
                                nn.Softmax(dim=1)
                                )
        self.critic_output_layers = nn.Sequential(
                                nn.Linear(layer_6, 1),
                                )
        # the universal network for all scheudling agents
        self.actor = nn.ModuleList([self.shared_FC_layers, self.actor_output_layers])
        self.critic = nn.ModuleList([self.shared_FC_layers, self.critic_output_layers])
        self.network = self.actor # only save the parameters of actor network
        # loss functions for critic, actor's "loss" is directly calculated
        self.critic_loss_func = F.smooth_l1_loss
        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    def actor_forward(self, x): # policy network
        #print('original',x)
        x = self.actor[0](x)
        prob = self.actor[1](x)
        #print('output',prob)
        return prob # returns the probability of action

    def critic_forward(self, x): # value network
        #print('original',x)
        x = self.critic[0](x)
        value = self.critic[1](x)
        #print('output',value)
        return value

class network_benchmark_policy(nn.Module):
    def __init__(self, input_size, output_size, **kwargs):
        super(network_benchmark_policy, self).__init__()
        print("*** Initialize ActorCritic networks ***")
        self.input_size = input_size
        self.output_size = output_size
        self.lr = 0.001
        self.flattened_input_size = torch.tensor(self.input_size).prod()
        # FCNN parameters
        layer_1 = 48
        layer_2 = 36
        layer_3 = 24
        layer_4 = 12
        # shared fully-connected layers
        self.shared_FC_layers = nn.Sequential(
                                nn.LayerNorm(self.input_size),
                                nn.Flatten(),
                                nn.Linear(self.flattened_input_size, layer_1),
                                nn.Tanh(),
                                nn.Linear(layer_1, layer_2),
                                nn.Tanh(),
                                nn.Linear(layer_2, layer_3),
                                nn.Tanh(),
                                nn.Linear(layer_3, layer_4),
                                nn.Tanh(),
                                )
        # output layer for actor and critic
        self.actor_output_layers = nn.Sequential(
                                nn.Linear(layer_4, self.output_size),
                                nn.Softmax(dim=1)
                                )
        self.critic_output_layers = nn.Sequential(
                                nn.Linear(layer_4, 1),
                                )
        # the universal network for all scheudling agents
        self.actor = nn.ModuleList([self.shared_FC_layers, self.actor_output_layers])
        self.critic = nn.ModuleList([self.shared_FC_layers, self.critic_output_layers])
        self.network = self.actor # only save the parameters of actor network
        # loss functions for critic, actor's "loss" is directly calculated
        self.critic_loss_func = F.smooth_l1_loss
        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    def actor_forward(self, x): # policy network
        #print('original',x)
        x = self.actor[0](x)
        prob = self.actor[1](x)
        #print('output',prob)
        return prob # returns the probability of action

    def critic_forward(self, x): # value network
        #print('original',x)
        x = self.critic[0](x)
        value = self.critic[1](x)
        #print('output',value)
        return value

'''
below are value-based encoder
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

class network_AS(nn.Module):
    def __init__(self, input_size, output_size):
        super(network_AS, self).__init__()
        self.lr = 0.001
        self.input_size = input_size
        self.output_size = output_size
        # for slicing the data
        self.no_size = 3
        self.pt_size = 6
        self.remaining_pt_size = 11
        self.ttd_slack_size = 16
        # FCNN parameters
        layer_1 = 48
        layer_2 = 36
        layer_3 = 36
        layer_4 = 24
        layer_5 = 24
        layer_6 = 12
        # normalization modules
        self.normlayer_no = nn.Sequential(
                                nn.InstanceNorm1d(3),
                                nn.Flatten()
                                )
        self.normlayer_pt = nn.Sequential(
                                nn.InstanceNorm1d(3),
                                nn.Flatten()
                                )
        self.normlayer_remaining_pt = nn.Sequential(
                                nn.InstanceNorm1d(5),
                                nn.Flatten()
                                )
        self.normlayer_ttd_slack = nn.Sequential(
                                nn.InstanceNorm1d(5),
                                nn.Flatten()
                                )
        # shared layers of machines
        self.subsequent_module = nn.Sequential(
                                nn.Linear(self.input_size, layer_1),
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
        self.network = nn.ModuleList([self.normlayer_no, self.normlayer_pt, self.normlayer_remaining_pt, self.normlayer_ttd_slack, self.subsequent_module])
        # accompanied by optimizer
        self.optimizer = optim.SGD(self.network.parameters(), lr=self.lr, momentum = 0.9)

    def forward(self, x, *args):
        #print('original',x)
        # slice the data
        x_no = x[:,:, : self.no_size]
        x_pt = x[:,:, self.no_size : self.pt_size]
        x_remaining_pt = x[:,:, self.pt_size : self.remaining_pt_size]
        x_ttd_slack = x[:,:, self.remaining_pt_size : self.ttd_slack_size]
        x_rest = x[:,:, self.ttd_slack_size :].squeeze(1)
        # normalize data in multiple channels
        x_normed_no = self.network[0](x_no)
        x_normed_pt = self.network[1](x_pt)
        x_normed_remaining_pt = self.network[2](x_remaining_pt)
        x_normed_ttd_slack = self.network[3](x_ttd_slack)
        #print('normalized',x_normed_no)
        # concatenate all data
        #print(x_normed_no, x_normed_pt, x_normed_remaining_pt, x_normed_ttd, x_normed_slack, x_rest)
        x = torch.cat([x_normed_no, x_normed_pt, x_normed_remaining_pt, x_normed_ttd_slack, x_rest], dim=1)
        #print('combined',x)
        # the last, independent part of module
        x = self.network[4](x)
        #print('output',x)
        return x

class network_TEST_AS(nn.Module):
    def __init__(self, input_size, output_size):
        super(network_TEST_AS, self).__init__()
        self.lr = 0.005
        self.input_size = input_size
        self.output_size = output_size
        # for slicing the data
        self.no_size = 3
        self.pt_size = 6
        self.remaining_pt_size = 11
        self.ttd_slack_size = 16
        # FCNN parameters
        layer_1 = 64
        layer_2 = 48
        layer_3 = 48
        layer_4 = 36
        layer_5 = 24
        layer_6 = 12
        # normalization modules
        self.normlayer_no = nn.Sequential(
                                nn.InstanceNorm1d(3),
                                nn.Flatten()
                                )
        self.normlayer_pt = nn.Sequential(
                                nn.InstanceNorm1d(3),
                                nn.Flatten()
                                )
        self.normlayer_remaining_pt = nn.Sequential(
                                nn.InstanceNorm1d(5),
                                nn.Flatten()
                                )
        self.normlayer_ttd_slack = nn.Sequential(
                                nn.InstanceNorm1d(5),
                                nn.Flatten()
                                )
        # shared layers of machines
        self.subsequent_module = nn.Sequential(
                                nn.Linear(self.input_size, layer_1),
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
        self.network = nn.ModuleList([self.normlayer_no, self.normlayer_pt, self.normlayer_remaining_pt, self.normlayer_ttd_slack, self.subsequent_module])
        # accompanied by optimizer
        self.optimizer = optim.SGD(self.network.parameters(), lr=self.lr, momentum = 0.9)

    def forward(self, x, *args):
        #print('original',x)
        # slice the data
        x_no = x[:,:, : self.no_size]
        x_pt = x[:,:, self.no_size : self.pt_size]
        x_remaining_pt = x[:,:, self.pt_size : self.remaining_pt_size]
        x_ttd_slack = x[:,:, self.remaining_pt_size : self.ttd_slack_size]
        x_rest = x[:,:, self.ttd_slack_size :].squeeze(1)
        # normalize data in multiple channels
        x_normed_no = self.network[0](x_no)
        x_normed_pt = self.network[1](x_pt)
        x_normed_remaining_pt = self.network[2](x_remaining_pt)
        x_normed_ttd_slack = self.network[3](x_ttd_slack)
        #print('normalized',x_normed_no)
        # concatenate all data
        #print(x_normed_no, x_normed_pt, x_normed_remaining_pt, x_normed_ttd, x_normed_slack, x_rest)
        x = torch.cat([x_normed_no, x_normed_pt, x_normed_remaining_pt, x_normed_ttd_slack, x_rest], dim=1)
        #print('combined',x)
        # the last, independent part of module
        x = self.network[4](x)
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
        layer_1 = 256
        layer_2 = 128
        layer_3 = 96
        layer_4 = 64
        layer_5 = 32
        layer_6 = 16
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
                                nn.Linear(layer_6, output_size),
                                )
        # Huber loss function
        self.loss_func = F.smooth_l1_loss
        # the universal network for all scheudling agents
        self.network = nn.ModuleList([self.norm_layer, self.FC_layers])
        # accompanied by optimizer
        self.optimizer = optim.SGD(self.network.parameters(), lr=self.lr, momentum = 0.9)
        #self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    def forward(self, x, *args):
        #print('original',x)
        x = self.network[0](x)
        x = self.network[1](x)
        #print('output',x)
        return x

class network_independent(nn.Module):
    def __init__(self, input_size, output_size, m_no):
        super(network_independent, self).__init__()
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
        self.MLP = nn.Sequential(
                                nn.LayerNorm(self.input_size),
                                nn.Flatten(),
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
        # the dictionary that stores ModuleList for each sequencing agent
        self.network_dict = {}
        self.optimizer_dict = {}
        for i in range(m_no):
            # for each agent, its module list contains shared and independent layers
            self.network_dict[i] = copy.deepcopy(self.MLP)
            # accompanied by an independent optimizer
            self.optimizer_dict[i] = optim.SGD(self.network_dict[i].parameters(), lr=self.lr, momentum = 0.9)

    def forward(self, x, m_idx):
        #print('original',x)
        x = self.network_dict[m_idx](x)
        #print('output',x)
        return x

class network_independent_AS(nn.Module):
    def __init__(self, input_size, output_size, m_no):
        super(network_independent_AS, self).__init__()
        self.lr = 0.005
        self.input_size = input_size
        self.output_size = output_size
        # for slicing the data
        self.no_size = 3
        self.pt_size = 6
        self.remaining_pt_size = 11
        self.ttd_slack_size = 16
        # FCNN parameters
        layer_1 = 64
        layer_2 = 48
        layer_3 = 48
        layer_4 = 36
        layer_5 = 24
        layer_6 = 12
        # normalization modules
        self.normlayer_no = nn.Sequential(
                                nn.InstanceNorm1d(3),
                                nn.Flatten()
                                )
        self.normlayer_pt = nn.Sequential(
                                nn.InstanceNorm1d(3),
                                nn.Flatten()
                                )
        self.normlayer_remaining_pt = nn.Sequential(
                                nn.InstanceNorm1d(5),
                                nn.Flatten()
                                )
        self.normlayer_ttd_slack = nn.Sequential(
                                nn.InstanceNorm1d(5),
                                nn.Flatten()
                                )
        # shared layers of machines
        self.subsequent_module = nn.Sequential(
                                nn.Linear(self.input_size, layer_1),
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
        # the dictionary that stores ModuleList for each sequencing agent
        self.network_dict = {}
        self.optimizer_dict = {}
        self.MLP = nn.ModuleList([self.normlayer_no, self.normlayer_pt, self.normlayer_remaining_pt, self.normlayer_ttd_slack, self.subsequent_module])
        for i in range(m_no):
            # for each agent, its module list contains shared and independent layers
            self.network_dict[i] = copy.deepcopy(self.MLP)
            # accompanied by an independent optimizer
            self.optimizer_dict[i] = optim.SGD(self.network_dict[i].parameters(), lr=self.lr, momentum = 0.9)

    def forward(self, x, m_idx):
        #print('original',x)
        # slice the data
        x_no = x[:,:, : self.no_size]
        x_pt = x[:,:, self.no_size : self.pt_size]
        x_remaining_pt = x[:,:, self.pt_size : self.remaining_pt_size]
        x_ttd_slack = x[:,:, self.remaining_pt_size : self.ttd_slack_size]
        x_rest = x[:,:, self.ttd_slack_size :].squeeze(1)
        # normalize data in multiple channels
        x_normed_no = self.network_dict[m_idx][0](x_no)
        x_normed_pt = self.network_dict[m_idx][1](x_pt)
        x_normed_remaining_pt = self.network_dict[m_idx][2](x_remaining_pt)
        x_normed_ttd_slack = self.network_dict[m_idx][3](x_ttd_slack)
        #print('normalized',x_normed_no)
        # concatenate all data
        #print(x_normed_no, x_normed_pt, x_normed_remaining_pt, x_normed_ttd, x_normed_slack, x_rest)
        x = torch.cat([x_normed_no, x_normed_pt, x_normed_remaining_pt, x_normed_ttd_slack, x_rest], dim=1)
        #print('combined',x)
        # the last, independent part of module
        x = self.network_dict[m_idx][4](x)
        #print('output',x)
        return x

class network_benchmark_value_based(nn.Module):
    def __init__(self, input_size, output_size):
        super(network_benchmark_value_based, self).__init__()
        self.lr = 0.005
        self.input_size = input_size
        self.output_size = output_size
        self.flattened_input_size = torch.tensor(self.input_size).prod()
        # FCNN parameters
        layer_1 = 32
        layer_2 = 24
        layer_3 = 12
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
                                nn.Linear(layer_3, output_size),
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
