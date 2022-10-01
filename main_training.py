import simpy
import sys
sys.path #sometimes need this to refresh the path
import matplotlib.pyplot as plt
import numpy as np

import Asset_machine as Machine
import Event_job_creation
import Event_breakdown_creation
import Brain_sequencing
import Validation

'''
Train deep MARL agents in simulation
'''

class shopfloor:
    def __init__(self,env,span,m_no,**kwargs):
        # STEP 1: create environment for simulation and control parameters
        self.env=env
        self.span = span
        self.m_no = m_no
        self.m_list = []

        # STEP 2: create instances of machines
        for i in range(m_no):
            expr1 = '''self.m_{} = Machine.machine(env, {}, print = 0)'''.format(i,i) # create machines
            exec(expr1)
            expr2 = '''self.m_list.append(self.m_{})'''.format(i) # add to machine list
            exec(expr2)

        # STEP 3: initialize the initial jobs, distribute jobs to workcenters
        # env, span, machine_list, workcenter_list, number_of_jobs, pt_range, due_tightness, E_utliz, print
        self.job_creator = Event_job_creation.creation\
        (self.env, self.span, self.m_list, [1,50], 3, 0.75, print = 0)
        self.job_creator.initial_output()

        # STEP 4: initialize all machines
        for i in range(m_no):
            expr3 = '''self.m_{}.initialization(self.m_list,self.job_creator)'''.format(i) # initialize all machines
            exec(expr3)

        #STEP 5: add a brain to the shop floor
        self.brain = Brain_sequencing.brain(self.env, self.job_creator, self.m_list, self.span/10, self.span,
            TEST = 0, reward_function = 1, bsf_start = 0)

    # FINAL STEP: start the simulaiton, and plot the loss/ reward record
    def simulation(self):
        self.env.run()
        self.brain.check_parameter()
        self.brain.loss_record_output(save=0)
        #self.brain.reward_record_output(save=0)



# create the environment instance for simulation
env = simpy.Environment()
span = 10000
scale = 10
show = True
# create the shop floor instance
# the command of startig the simulation is included in shopfloor instance, run till there's no more events
spf = shopfloor(env, span, scale,)
spf.simulation()
