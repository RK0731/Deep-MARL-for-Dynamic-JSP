import numpy as np
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from src import brain_sequencing
from torch.distributions import Categorical
from tabulate import tabulate

'''
This module is used for applying trained parameters in the experiments
'''

class DRL_sequencing(brain_sequencing.brain):
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
        if 'validated' in kwargs and kwargs['validated']:
            print("---> Validated Mode ON <---")
            self.model_path = Path.cwd()/"trained_models"/"validated.pt"
            # adaptive input size
            self.input_size = self.state_direct(self.m_list[0].sequencing_data_generation()).size()
            self.input_size_as_list = list(self.input_size)
            self.output_size = 4
            self.network = brain_sequencing.network_value_based(self.input_size, self.output_size)
            self.network.network.load_state_dict(torch.load(self.model_path))
            self.network.eval()  # must have this if you're loading a model, unnecessray for loading state_dict
            for m in self.m_list:
                m.job_sequencing = self.action_direct
                self.build_state = self.state_direct
        elif 'TEST' in kwargs and kwargs['TEST']:
            print("---!!! TEST mode ON !!!---")
            self.model_path = Path.cwd()/"trained_models"/f"TEST_DDQN_rwd{kwargs['reward_function']}.pt"
            # adaptive input size
            self.input_size = self.state_direct(self.m_list[0].sequencing_data_generation()).size()
            self.input_size_as_list = list(self.input_size)
            self.output_size = 4
            self.network = brain_sequencing.network_TEST(self.input_size, self.output_size)
            self.network.network.load_state_dict(torch.load(self.model_path))
            self.network.eval()  # must have this if you're loading a model, unnecessray for loading state_dict
            for m in self.m_list:
                m.job_sequencing = self.action_direct
                self.build_state = self.state_direct
        elif 'import_from' in kwargs and kwargs['import_from']:
            print("---> VALIDATION MODE <---")
            self.model_path = Path.cwd()/f"trained_models{kwargs['import_from']}.pt"
            self.input_size = self.state_direct(self.m_list[0].sequencing_data_generation()).size()
            self.output_size = 4
            self.network = brain_sequencing.network_value_based(self.input_size, self.output_size)
            self.network.network.load_state_dict(torch.load(self.model_path))
            self.network.eval()  # must have this if you're loading a model, unnecessray for loading state_dict
            for m in self.m_list:
                m.job_sequencing = self.action_direct
                self.build_state = self.state_direct
        else:
            print("---X DEFAULT (DDQN) mode ON X---")
            self.model_path = Path.cwd()/"trained_models"/f"DDQN_rwd{kwargs['reward_function']}.pt"
            self.input_size = self.state_direct(self.m_list[0].sequencing_data_generation()).size()
            self.input_size_as_list = list(self.input_size)
            self.output_size = 4
            self.network = brain_sequencing.network_value_based(self.input_size, self.output_size)
            self.network.network.load_state_dict(torch.load(self.model_path))
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
        print("Collect from:",self.model_path)
        print('Trained with Rwd.Func.:',self.kwargs['reward_function'])
        print('State function:',self.build_state.__name__)
        print('ANN architecture:',self.network.__class__.__name__)
        print('------------------Scenario Check ------------------')
        print("PT heterogeneity:",self.job_creator.pt_range)
        print('Due date tightness:',self.job_creator.tightness)
        print('Utilization rate:',self.job_creator.E_utliz)
        print('----------------------------------------------------------------------')
