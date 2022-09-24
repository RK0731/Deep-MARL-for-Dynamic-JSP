import numpy as np
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import Brain_sequencing
from torch.distributions import Categorical
from tabulate import tabulate
import Rule_sequencing

class DRL_sequencing(Brain_sequencing.brain): # inherit a bunch of functions from brain class
    def __init__(self, env, machine_list, job_creator, span, *args, **kwargs):
        # initialize the environment and the workcenter to be controlled
        self.env = env
        # get list of alll machines, for collecting the global data
        self.m_list = machine_list
        self.job_creator = job_creator
        self.kwargs = kwargs
        '''
        choose the trained parameters by its reward function
        '''
        if 'reward_function' in kwargs:
            pass
        else:
            print('WARNING: reward function is not specified')
            raise Exception
        # build action NN for each target machine
        if 'A2C' in kwargs and kwargs['A2C']:
            print("---> A2C MODE ON <---")
            self.address_seed = "{}\\trained_models\\A2C_rwd" + str(kwargs['reward_function']) + ".pt"
            self.input_size = self.state_direct(self.m_list[0].sequencing_data_generation()).size()
            self.input_size_as_list = list(self.input_size)
            self.output_size = 4
            self.network = Brain_sequencing.network_ActorCritic(self.input_size, self.output_size)
            self.network.actor.load_state_dict(torch.load(self.address_seed.format(sys.path[0])))
            self.network.eval()  # must have this if you're loading a model, unnecessray for loading state_dict
            for m in self.m_list:
                m.job_sequencing = self.action_A2C
                self.build_state = self.state_direct
        elif 'bsf_A2C' in kwargs and kwargs['bsf_A2C']:
            print("---> BSF A2C ON <---")
            self.address_seed = "{}\\trained_models\\bsf_A2C.pt"
            self.input_size = self.state_direct(self.m_list[0].sequencing_data_generation()).size()
            self.output_size = 4
            self.network = Brain_sequencing.network_ActorCritic(self.input_size, self.output_size)
            self.network.actor.load_state_dict(torch.load(self.address_seed.format(sys.path[0])))
            self.network.eval()  # must have this if you're loading a model, unnecessray for loading state_dict
            for m in self.m_list:
                m.job_sequencing = self.action_A2C
                self.build_state = self.state_direct
        elif 'bsf_DDQN' in kwargs and kwargs['bsf_DDQN']:
            print("---> BSF DDQN ON <---")
            self.address_seed = "{}/trained_models/bsf_DDQN.pt"
            # adaptive input size
            self.input_size = self.state_direct(self.m_list[0].sequencing_data_generation()).size()
            self.input_size_as_list = list(self.input_size)
            self.output_size = 4
            self.network = Brain_sequencing.network_value_based(self.input_size, self.output_size)
            self.network.network.load_state_dict(torch.load(self.address_seed.format(sys.path[0])))
            self.network.eval()  # must have this if you're loading a model, unnecessray for loading state_dict
            for m in self.m_list:
                m.job_sequencing = self.action_direct
                self.build_state = self.state_direct
        elif 'bsf_TEST' in kwargs and kwargs['bsf_TEST']:
            print("---> BSF TEST ON <---")
            self.address_seed = "{}\\trained_models\\bsf_TEST.pt"
            # adaptive input size
            self.input_size = self.state_direct(self.m_list[0].sequencing_data_generation()).size()
            self.input_size_as_list = list(self.input_size)
            self.output_size = 4
            self.network = Brain_sequencing.network_TEST(self.input_size, self.output_size)
            self.network.network.load_state_dict(torch.load(self.address_seed.format(sys.path[0])))
            self.network.eval()  # must have this if you're loading a model, unnecessray for loading state_dict
            for m in self.m_list:
                m.job_sequencing = self.action_direct
                self.build_state = self.state_direct
        elif 'TEST' in kwargs and kwargs['TEST']:
            print("---!!! TEST mode ON !!!---")
            self.address_seed = "{}\\trained_models\\TEST_DDQN_rwd"+str(kwargs['reward_function'])+".pt"
            # adaptive input size
            self.input_size = self.state_direct(self.m_list[0].sequencing_data_generation()).size()
            self.input_size_as_list = list(self.input_size)
            self.output_size = 4
            self.network = Brain_sequencing.network_TEST(self.input_size, self.output_size)
            self.network.network.load_state_dict(torch.load(self.address_seed.format(sys.path[0])))
            self.network.eval()  # must have this if you're loading a model, unnecessray for loading state_dict
            for m in self.m_list:
                m.job_sequencing = self.action_direct
                self.build_state = self.state_direct
        elif 'DDQN_SI' in kwargs and kwargs['DDQN_SI']:
            print("---> SI mode ON <---")
            self.address_seed = "{}\\trained_models\\DDQN_SI_rwd"+str(kwargs['reward_function'])+".pt"
            # adaptive input size
            self.input_size = self.state_direct(self.m_list[0].sequencing_data_generation()).size()
            self.input_size_as_list = list(self.input_size)
            self.output_size = 5
            self.network = Brain_sequencing.network_value_based(self.input_size, self.output_size)
            self.network.network.load_state_dict(torch.load(self.address_seed.format(sys.path[0])))
            self.network.eval()  # must have this if you're loading a model, unnecessray for loading state_dict
            for m in self.m_list:
                m.job_sequencing = self.action_direct_SI
                self.build_state = self.state_direct
        elif 'IQL' in kwargs and kwargs['IQL']: # baseline, independent DQN agents
            print("---> Baseline ON <---")
            self.address_seed = "{}\\trained_models\\independent\\IQL_rwd"+str(kwargs['reward_function'])+"_{}.pt"
            # adaptive input size
            self.input_size = self.state_direct(self.m_list[0].sequencing_data_generation()).size()
            self.input_size_as_list = list(self.input_size)
            self.output_size = 4
            self.network = Brain_sequencing.network_independent(self.input_size, self.output_size, len(self.m_list))
            for m in self.m_list:
                self.network.network_dict[m.m_idx].load_state_dict(torch.load(self.address_seed.format(sys.path[0], m.m_idx)))
            self.network.eval()  # must have this if you're loading a model, unnecessray for loading state_dict
            for m in self.m_list:
                m.job_sequencing = self.action_direct
                self.build_state = self.state_direct
        elif 'I_DDQN' in kwargs and kwargs['I_DDQN']: # baseline, independent DQN agents
            print("---> Independent Double DQN ON <---")
            self.address_seed = "{}\\trained_models\\independent\\I_DDQN_rwd"+str(kwargs['reward_function'])+"_{}.pt"
            # adaptive input size
            self.input_size = self.state_direct(self.m_list[0].sequencing_data_generation()).size()
            self.input_size_as_list = list(self.input_size)
            self.output_size = 4
            self.network = Brain_sequencing.network_independent(self.input_size, self.output_size, len(self.m_list))
            for m in self.m_list:
                self.network.network_dict[m.m_idx].load_state_dict(torch.load(self.address_seed.format(sys.path[0], m.m_idx)))
            self.network.eval()  # must have this if you're loading a model, unnecessray for loading state_dict
            for m in self.m_list:
                m.job_sequencing = self.action_direct
                self.build_state = self.state_direct
        elif 'import_from' in kwargs and kwargs['import_from']:
            print("---> VALIDATION MODE <---")
            self.address_seed = "{}\\trained_models\\" + str(kwargs['import_from']) + ".pt"
            self.input_size = self.state_direct(self.m_list[0].sequencing_data_generation()).size()
            self.output_size = 4
            self.network = Brain_sequencing.network_value_based(self.input_size, self.output_size)
            self.network.network.load_state_dict(torch.load(self.address_seed.format(sys.path[0])))
            self.network.eval()  # must have this if you're loading a model, unnecessray for loading state_dict
            for m in self.m_list:
                m.job_sequencing = self.action_direct
                self.build_state = self.state_direct
        elif 'TEST_AS' in kwargs and kwargs['TEST_AS']:
            print("---> VALIDATION MODE <---")
            self.address_seed = "{}\\trained_models\\TEST_AS_rwd"+str(kwargs['reward_function'])+".pt"
            self.input_size =  len(self.state_multi_channel(self.m_list[0].sequencing_data_generation()))
            self.input_size_as_list = [1,self.input_size]
            self.func_list = [Rule_sequencing.SPT, Rule_sequencing.WINQ, Rule_sequencing.MS, Rule_sequencing.CR]
            self.output_size = 4
            self.network = Brain_sequencing.network_TEST_AS(self.input_size, self.output_size)
            self.network.network.load_state_dict(torch.load(self.address_seed.format(sys.path[0])))
            self.network.eval()  # must have this if you're loading a model, unnecessray for loading state_dict
            for m in self.m_list:
                m.job_sequencing = self.action_AS
                self.build_state = self.state_multi_channel
        elif 'AS' in kwargs and kwargs['AS']:
            print("---> VALIDATION MODE <---")
            self.address_seed = "{}\\trained_models\\Abstracted_state_rwd"+str(kwargs['reward_function'])+".pt"
            self.input_size =  len(self.state_multi_channel(self.m_list[0].sequencing_data_generation()))
            self.input_size_as_list = [1,self.input_size]
            self.func_list = [Rule_sequencing.SPT, Rule_sequencing.WINQ, Rule_sequencing.MS, Rule_sequencing.CR]
            self.output_size = 4
            self.network = Brain_sequencing.network_AS(self.input_size, self.output_size)
            self.network.network.load_state_dict(torch.load(self.address_seed.format(sys.path[0])))
            self.network.eval()  # must have this if you're loading a model, unnecessray for loading state_dict
            for m in self.m_list:
                m.job_sequencing = self.action_AS
                self.build_state = self.state_multi_channel
        elif 'IQL_AS' in kwargs and kwargs['IQL_AS']: # baseline, independent DQN agents
            print("---> IQL_AS ON <---")
            self.address_seed = "{}\\trained_models\\independent\\IQL_AS_rwd"+str(kwargs['reward_function'])+"_{}.pt"
            self.input_size =  len(self.state_multi_channel(self.m_list[0].sequencing_data_generation()))
            self.input_size_as_list = [1,self.input_size]
            self.func_list = [Rule_sequencing.SPT, Rule_sequencing.WINQ, Rule_sequencing.MS, Rule_sequencing.CR]
            self.output_size = 4
            self.network = Brain_sequencing.network_independent_AS(self.input_size, self.output_size, len(self.m_list))
            for m in self.m_list:
                self.network.network_dict[m.m_idx].load_state_dict(torch.load(self.address_seed.format(sys.path[0], m.m_idx)))
            self.network.eval()  # must have this if you're loading a model, unnecessray for loading state_dict
            for m in self.m_list:
                m.job_sequencing = self.action_AS
                self.build_state = self.state_multi_channel
        elif 'I_DDQN_AS' in kwargs and kwargs['I_DDQN_AS']: # baseline, independent DQN agents
            print("---> I_DDQN_AS ON <---")
            self.address_seed = "{}\\trained_models\\independent\\I_DDQN_AS_rwd"+str(kwargs['reward_function'])+"_{}.pt"
            self.input_size =  len(self.state_multi_channel(self.m_list[0].sequencing_data_generation()))
            self.input_size_as_list = [1,self.input_size]
            self.func_list = [Rule_sequencing.SPT, Rule_sequencing.WINQ, Rule_sequencing.MS, Rule_sequencing.CR]
            self.output_size = 4
            self.network = Brain_sequencing.network_independent_AS(self.input_size, self.output_size, len(self.m_list))
            for m in self.m_list:
                self.network.network_dict[m.m_idx].load_state_dict(torch.load(self.address_seed.format(sys.path[0], m.m_idx)))
            self.network.eval()  # must have this if you're loading a model, unnecessray for loading state_dict
            for m in self.m_list:
                m.job_sequencing = self.action_AS
                self.build_state = self.state_multi_channel
        else:
            print("---X DEFAULT (DDQN) mode ON X---")
            self.address_seed = "{}\\trained_models\\DDQN_rwd"+str(kwargs['reward_function'])+".pt"
            self.input_size = self.state_direct(self.m_list[0].sequencing_data_generation()).size()
            self.input_size_as_list = list(self.input_size)
            self.output_size = 4
            self.network = Brain_sequencing.network_value_based(self.input_size, self.output_size)
            self.network.network.load_state_dict(torch.load(self.address_seed.format(sys.path[0])))
            self.network.eval()  # must have this if you're loading a model, unnecessray for loading state_dict
            self.build_state = self.state_direct
            for m in self.m_list:
                m.job_sequencing = self.action_direct

        print('--------------------------')
        #print("Dictionary of networks:\n",self.net_dict)
        # check if need to show the specific selection
        self.show = False
        if 'show' in kwargs and kwargs['show']:
            self.show = True

    '''action functions, different from brain module'''
    def action_direct(self, sqc_data):
        s_t = self.build_state(sqc_data)
        m_idx = sqc_data[-1]
        # input state to action network, produce the state-action value
        value = self.network.forward(s_t.reshape([1]+self.input_size_as_list),m_idx)
        # greedy policy
        a_t = torch.argmax(value)
        self.strategic_idleness_bool = False # no strategic idleness
        if self.show:
            print(value,a_t)
        #print('convert action to', a_t)
        job_position, j_idx = self.action_conversion(a_t)
        return job_position, self.strategic_idleness_bool, self.strategic_idleness_time

    def action_direct_SI(self, sqc_data): # direct selection that allows strategic idleness
        s_t = self.build_state(sqc_data)
        m_idx = sqc_data[-1]
        # input state to action network, produce the state-action value
        value = self.network.forward(s_t.reshape([1]+self.input_size_as_list),m_idx).squeeze()
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
        if self.show:
            print(value,a_t)
        job_position, j_idx = self.action_conversion(a_t)
        return job_position, self.strategic_idleness_bool, self.strategic_idleness_time

    def action_A2C(self, sqc_data):
        s_t = self.build_state(sqc_data)
        m_idx = sqc_data[-1]
        # input state to action network, produce the state-action value
        prob = self.network.actor_forward(s_t.reshape([1]+self.input_size_as_list))
        # stochastic policy
        a_t = Categorical(prob).sample()
        self.strategic_idleness_bool = False # then no need do strategic idleness
        if self.show:
            print('A2C Action: %s'%(prob,a_t))
        job_position, j_idx = self.action_conversion(a_t)
        return job_position, self.strategic_idleness_bool, self.strategic_idleness_time

    def action_sqc_rule_value_based(self, sqc_data):
        s_t = self.build_state(sqc_data)
        m_idx = sqc_data[-1]
        value = self.network.forward(s_t.reshape([1]+self.input_size_as_list),m_idx).squeeze()
        a_t = torch.argmax(value)
        self.strategic_idleness_bool = False
        rule_a_t = self.func_list[a_t](s_t)
        job_position, j_idx = self.action_conversion(rule_a_t)
        return job_position, self.strategic_idleness_bool, self.strategic_idleness_time

    def action_sqc_rule_policy(self, sqc_data):
        s_t = self.build_state(sqc_data)
        m_idx = sqc_data[-1]
        prob = self.network.actor_forward(s_t.reshape([1]+self.input_size_as_list)).squeeze()
        a_t = Categorical(prob).sample()
        self.strategic_idleness_bool = False
        rule_a_t = self.func_list[a_t](s_t)
        job_position, j_idx = self.action_conversion(rule_a_t)
        return job_position, self.strategic_idleness_bool, self.strategic_idleness_time

    def action_AS(self, sqc_data):
        s_t = self.build_state(sqc_data)
        m_idx = sqc_data[-1]
        # input state to policy network, produce the state-action value
        value = self.network.forward(s_t.reshape([1,1,self.input_size]),m_idx)
        # greedy policy
        a_t = torch.argmax(value)
        #print('Sequencing NN, action %s / By Brain'%(a_t))
        job_position, self.strategic_idleness_bool, self.strategic_idleness_time = self.func_list[a_t](sqc_data)
        return job_position, self.strategic_idleness_bool, self.strategic_idleness_time

    def check_parameter(self):
        print('------------------ Sequencing Brain Parameter Check ------------------')
        print("Collect from:",self.address_seed)
        print('Trained with Rwd.Func.:',self.kwargs['reward_function'])
        print('State function:',self.build_state.__name__)
        print('ANN architecture:',self.network.__class__.__name__)
        print('------------------Scenario Check ------------------')
        print("PT heterogeneity:",self.job_creator.pt_range)
        print('Due date tightness:',self.job_creator.tightness)
        print('Utilization rate:',self.job_creator.E_utliz)
        print('----------------------------------------------------------------------')
