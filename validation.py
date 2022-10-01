import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import Brain_sequencing
from torch.distributions import Categorical
from tabulate import tabulate
import Rule_sequencing

'''
This module is used for applying trained parameters in the experiments
'''

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
        if 'bsf_DDQN' in kwargs and kwargs['bsf_DDQN']:
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

    '''action function, direct selection of job'''
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

    ''' after experiment, see if it's the intended scenario'''
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
