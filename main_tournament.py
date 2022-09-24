import simpy
import sys
sys.path #sometimes need this to refresh the path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import numpy as np
from tabulate import tabulate
import pandas as pd
from pandas import DataFrame

import Asset_machine as Machine
import Event_job_creation
import Event_breakdown_creation
import heterogeneity_creation
import validation

class shopfloor:
    def __init__(self,env,span,m_no,**kwargs):
        # STEP 1: create environment instances and specifiy simulation span
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
        if 'seed' in kwargs:
            self.job_creator = Event_job_creation.creation\
            (self.env, self.span, self.m_list, [1,50], 3, 0.7, seed=kwargs['seed'])
            #self.job_creator.output()
        else:
            print("WARNING: seed is not fixed !!")
            raise Exception

        # STEP 4: initialize all machines
        for i in range(m_no):
            expr3 = '''self.m_{}.initialization(self.m_list,self.job_creator)'''.format(i) # initialize all machines
            exec(expr3)

        # STEP 6: initialize the scenario creator
        '''
        intervals = np.ones(8)*self.span/8
        pt_range_list = [[2,11],[5,8],[3,12],[6,8],[4,8],[3,12],[2,8],[5,8]]
        self.scenario = heterogeneity_creation.creation(self.env, self.job_creator, intervals, pt_range_list)
        '''

        # specify the architecture of DRL
        if 'sequencing_rule' in kwargs:
            print("Taking over: machines use {} sequencing rule".format(kwargs['sequencing_rule']))
            for m in self.m_list:
                order = "m.job_sequencing = sequencing." + kwargs['sequencing_rule']
                try:
                    exec(order)
                except:
                    print("Rule assigned to machine {} is invalid !".format(m.m_label))
                    raise Exception
        elif len(kwargs):
            arch = kwargs['arch'] + "=True"
            if type(kwargs['rwd_func']) is str:
                rwd_func = "reward_function='{}'".format(kwargs['rwd_func'])
            else:
                rwd_func = 'reward_function=' + str(kwargs['rwd_func'])
            order = "self.sequencing_brain = validation.DRL_sequencing(self.env, self.m_list, self.job_creator, self.span, {}, {})".format(arch,rwd_func)
            exec(order)
            print("---> {},{} <---".format(arch,rwd_func))

    def simulation(self):
        self.env.run()

# dictionary to store shopfloors and production record
spf_dict = {}
production_record = {}

'''
arch_set = ['IQL','I_DDQN','I_DDQN','Default','N_2019','U_2020','IQL','bsf_DDQN',]
func_set = [12,12,3,12,3,3,11,'',]
'''
'''
arch_set = ['IQL','TEST','TEST','Default','bsf_DDQN']
func_set = [12,12,3,12,'']
'''
arch_set = ['test','bsf_DDQN']
func_set = [3,'']

title = [x+str(func_set[i]) for i,x in enumerate(arch_set)]

# how long is the simulation
span = 1000
scale = 10
sum_record = []
benchmark_record = []
max_record = []
rate_record = []
iteration = 20
FIFO = 0
export_result = 0

if FIFO:
    title.insert(0,'FIFO')
if export_result:
    title = ['I-DQN','I-DDQN','I-DDQN-RS','P-DDQN','N-2019','U-2020','O-2021','deep MARL-RS',]


for run in range(iteration):
    print('******************* ITERATION-{} *******************'.format(run))
    sum_record.append([])
    benchmark_record.append([])
    max_record.append([])
    rate_record.append([])
    seed = np.random.randint(2000000000)
    # run simulation with different rules
    if FIFO:
        env = simpy.Environment()
        spf = shopfloor(env, span, scale, sequencing_rule = 'FIFO', seed = seed)
        spf.simulation()
        output_time, cumulative_tard, tard_mean, tard_max, tard_rate = spf.job_creator.tardiness_output()
        sum_record[run].append(cumulative_tard[-1])
        max_record[run].append(tard_max)
        rate_record[run].append(tard_rate)
    for idx,x in enumerate(arch_set):
        # and extra run with DRL
        env = simpy.Environment()
        spf = shopfloor(env, span, scale, arch = x, rwd_func = func_set[idx], seed = seed)
        spf.simulation()
        output_time, cumulative_tard, tard_mean, tard_max, tard_rate = spf.job_creator.tardiness_output()
        sum_record[run].append(cumulative_tard[-1])
        max_record[run].append(tard_max)
        rate_record[run].append(tard_rate)
        #print('Number of jobs created',spf.job_creator.total_no)


#print(sum_record)
print('-------------- Complete Record --------------')
print(tabulate(sum_record, headers=title))
print('-------------- Average Performance --------------')

# get the overall performance (include DRL)
avg = np.mean(sum_record,axis=0)
max = np.mean(max_record,axis=0)
tardy_rate = np.around(np.mean(rate_record,axis=0)*100,2)
ratio = np.around(avg/avg.min()*100,2)
rank = np.argsort(ratio)
winning_rate = np.zeros(len(title))
for idx in np.argmin(sum_record,axis=1):
    winning_rate[idx] += 1
winning_rate = np.around(winning_rate/iteration*100,2)
for rank,rule in enumerate(rank):
    print("{}, avg.: {} | max: {} | %: {}% | tardy %: {}% | wining rate: {}%"\
    .format(title[rule],avg[rule],max[rule],ratio[rule],tardy_rate[rule],winning_rate[rule]))

# check the parameter and scenario setting
spf.sequencing_brain.check_parameter()

if export_result:
    df_win_rate = DataFrame([winning_rate], columns=title)
    #print(df_win_rate)
    df_sum = DataFrame(sum_record, columns=title)
    #print(df_sum)
    df_tardy_rate = DataFrame(rate_record, columns=title)
    #print(df_tardy_rate)
    df_max = DataFrame(max_record, columns=title)
    #print(df_max)
    address = sys.path[0]+'\\experiment_result_figure\\RAW_tournament.xlsx'
    Excelwriter = pd.ExcelWriter(address,engine="xlsxwriter")
    dflist = [df_win_rate, df_sum, df_tardy_rate, df_max]
    sheetname = ['win rate','sum', 'tardy rate', 'maximum']

    for i,df in enumerate(dflist):
        df.to_excel(Excelwriter, sheet_name=sheetname[i], index=False)
    Excelwriter.save()
    print('export to {}'.format(address))
