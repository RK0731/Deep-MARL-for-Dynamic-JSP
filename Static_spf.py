import simpy
import sys
import torch
import numpy as np
from tabulate import tabulate
import pandas as pd
from pandas import DataFrame

import Rule_sequencing as Sequencing
import Asset_machine as Machine
import Static_job_creation
import Validation

'''
Shop floor
'''

class shopfloor:
    def __init__(self, env, operation_sequence, processing_time, due_date, **kwargs):
        # STEP 1: create environment for simulation and control parameters
        self.env=env
        self.m_no = len(set(np.concatenate(operation_sequence)))
        self.m_list = []

        # STEP 2: create instances of machines
        for i in range(self.m_no):
            expr1 = '''self.m_{} = Machine.machine(env, {}, print = 0)'''.format(i,i) # create machines
            exec(expr1)
            expr2 = '''self.m_list.append(self.m_{})'''.format(i) # add to machine list
            exec(expr2)

        # STEP 3: initialize the initial jobs, distribute jobs to machines
        self.job_creator = Static_job_creation.creation\
        (self.env, self.m_list, operation_sequence, processing_time, due_date)


        # STEP 4: initialize all machines
        for i in range(self.m_no):
            expr3 = '''self.m_{}.initialization(self.m_list,self.job_creator)'''.format(i) # initialize all machines
            exec(expr3)

        # STEP 5: set sequencing or routing rules, and DRL
        # check if need to reset sequencing rule
        if 'sequencing_rule' in kwargs:
            print(" machines use {} sequencing rule".format(kwargs['sequencing_rule']))
            for m in self.m_list:
                order = "m.job_sequencing = Sequencing." + kwargs['sequencing_rule']
                try:
                    exec(order)
                except:
                    print("Rule assigned to machine {} is invalid !".format(m.m_label))
                    raise Exception

        # specify the architecture of DRL
        if 'MR' in kwargs and kwargs['MR']:
            print("---> Minimal Repetition mode ON <---")
            self.sequencing_brain = Validation.DRL_sequencing(self.env, self.m_list, self.job_creator, 100, \
            bsf_DDQN = 1, show = 0, reward_function = 3 )


    def simulation(self):
        self.env.run()
