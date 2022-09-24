import simpy
import sys
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
import pandas as pd
from pandas import DataFrame

import Asset_machine as Machine
import Rule_sequencing as Sequencing
import Event_job_creation
import validation

'''
Run the simulation and compare the result with other priority rules
'''

class shopfloor:
    def __init__(self, env, span, m_no, **kwargs):
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
        if 'seed' in kwargs:
            self.job_creator = Event_job_creation.creation\
            (self.env, self.span, self.m_list, [1,50], 3, 0.9, seed=kwargs['seed'])
            #self.job_creator.initial_output()
        else:
            print("WARNING: seed is not fixed !!")
            raise Exception

        # STEP 4: initialize all machines
        for i in range(m_no):
            expr3 = '''self.m_{}.initialization(self.m_list,self.job_creator)'''.format(i) # initialize all machines
            exec(expr3)

        # STEP 5: set sequencing or routing rules, and DRL
        if 'sequencing_rule' in kwargs:
            print("Taking over: machines use {} sequencing rule".format(kwargs['sequencing_rule']))
            for m in self.m_list:
                order = "m.job_sequencing = Sequencing." + kwargs['sequencing_rule']
                try:
                    exec(order)
                except:
                    print("Rule assigned to machine {} is invalid !".format(m.m_label))
                    raise Exception

        # if DRL is used in this run
        if 'DRL' in kwargs and kwargs['DRL']:
            print("---> DRL Sequencing mode ON <---")
            self.sequencing_brain = validation.DRL_sequencing(self.env, self.m_list, self.job_creator, self.span, \
            TEST = 0, bsf_DDQN = 1, show = 0,  reward_function = 1)


    def simulation(self):
        self.env.run()


# dictionary to store shopfloors and production record
spf_dict = {}
production_record = {}
# list of benchmarks 
benchmark = ['FIFO','PTWINQS','DPTLWKRS','MDD','MOD']
title = benchmark + ['deep MARL-RS']
# experiment settings
span = 1000
scale = 10
sum_record = []
benchmark_record = []
max_record = []
rate_record = []
iteration = 10
export_result = 0

for run in range(iteration):
    print('******************* ITERATION-{} *******************'.format(run))
    sum_record.append([])
    benchmark_record.append([])
    max_record.append([])
    rate_record.append([])
    seed = np.random.randint(2000000000)
    # run simulation with different rules
    for idx,rule in enumerate(benchmark):
        # create the environment instance for simulation
        env = simpy.Environment()
        spf = shopfloor(env, span, scale, sequencing_rule = rule, seed = seed)
        spf.simulation()
        output_time, cumulative_tard, tard_mean, tard_max, tard_rate = spf.job_creator.tardiness_output()
        sum_record[run].append(cumulative_tard[-1])
        benchmark_record[run].append(cumulative_tard[-1])
        max_record[run].append(tard_max)
        rate_record[run].append(tard_rate)
    # an extra run with DRL
    env = simpy.Environment()
    spf = shopfloor(env, span, scale, DRL = True, seed = seed)
    spf.simulation()
    output_time, cumulative_tard, tard_mean, tard_max, tard_rate = spf.job_creator.tardiness_output()
    sum_record[run].append(cumulative_tard[-1])
    max_record[run].append(tard_max)
    rate_record[run].append(tard_rate)
    print('Number of jobs created',spf.job_creator.total_no)

print('-------------- Complete Record --------------')
print(tabulate(sum_record, headers=title))
print('-------------- Average Performance --------------')

# get the performnce without DRL
avg_b = np.mean(benchmark_record,axis=0)
ratio_b = np.around(avg_b/avg_b.max()*100,2)
winning_rate_b = np.zeros(len(title))
for idx in np.argmin(benchmark_record,axis=1):
    winning_rate_b[idx] += 1
winning_rate_b = np.around(winning_rate_b/iteration*100,2)

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
    print("{}, avg.: {} | max: {} | %: {}% | tardy %: {}% | winning rate: {}/{}%"\
    .format(title[rule],avg[rule],max[rule],ratio[rule],tardy_rate[rule],winning_rate_b[rule],winning_rate[rule]))

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
    df_before_win_rate = DataFrame([winning_rate_b], columns=title)
    address = sys.path[0]+'\\Thesis_figures\\RAW_experiment.xlsx'
    Excelwriter = pd.ExcelWriter(address,engine="xlsxwriter")
    dflist = [df_win_rate, df_sum, df_tardy_rate, df_max, df_before_win_rate]
    sheetname = ['win rate','sum', 'tardy rate', 'maximum','before win rate']

    for i,df in enumerate(dflist):
        df.to_excel(Excelwriter, sheet_name=sheetname[i], index=False)
    Excelwriter.save()
    print('export to {}'.format(address))
