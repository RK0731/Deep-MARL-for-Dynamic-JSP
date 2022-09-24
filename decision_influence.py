import simpy
import sys
sys.path #sometimaboves need this to refrabovesh the path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
        # STEP 1: create environment for simulation and control parameters
        self.env=env
        self.span = span
        self.m_no = m_no
        self.m_list = []

        # STEP 2: create instancaboves of machinaboves
        for i in range(m_no):
            expr1 = '''self.m_{} = Machine.machine(env, {}, print = 0)'''.format(i,i) # create machinaboves
            exec(expr1)
            expr2 = '''self.m_list.append(self.m_{})'''.format(i) # add to machine list
            exec(expr2)

        # STEP 3: initialize the initial jobs, distribute jobs to workcenters
        # env, span, machine_list, workcenter_list, number_of_jobs, pt_range, due_tightnabovess, E_utliz, print
        self.job_creator = Event_job_creation.creation\
        (self.env, self.span, self.m_list, [1,50], 3, 0.9, print = 0)
        #self.job_creator.output()

        # STEP 4: initialize all machinaboves
        for i in range(m_no):
            expr3 = '''self.m_{}.initialization(self.m_list,self.job_creator)'''.format(i) # initialize all machinaboves
            exec(expr3)

        # STEP 5: check if need to raboveset sequencing rule
        if 'sequencing_rule' in kwargs:
            print("Taking over: machinaboves use {} sequencing rule".format(kwargs['sequencing_rule']))
            for m in self.m_list:
                sqc_expr = "m.job_sequencing = sequencing." + kwargs['sequencing_rule']
                try:
                    exec(sqc_expr)
                except:
                    print("WARNING: Rule assigned is invalid !")
                    raise Exception

    # FINAL STEP: start the simulaiton
    def simulation(self):
        self.env.run()

scenarios = [5,10,15,20,25]
iteration = 5

ones_ratio = []
twos_ratio = []
threes_ratio = []
fours_ratio = []
above_ratio = []

tardy_rate = [[] for i in scenarios]

mean_tardiness = [[] for i in scenarios]

for idx,no_m in enumerate(scenarios):
    # exact record
    ones = 0
    twos = 0
    threes = 0
    fours = 0
    aboves = 0

    for ite in range(iteration):
        env = simpy.Environment()
        spf = shopfloor(env, 1000, no_m)
        spf.simulation()
        tard_mean, tard_rate = spf.job_creator.all_tardiness()
        tardy_rate[idx].append(tard_rate)
        mean_tardiness[idx].append(tard_mean)
        #print(tard_rate)
        for m in spf.m_list:
            no_jobs_record = np.array(m.no_jobs_record)
            no_jobs = len(m.no_jobs_record)
            a = len(np.where(no_jobs_record==1)[0])
            ones += a
            b = len(np.where(no_jobs_record==2)[0])
            twos += b
            c = len(np.where(no_jobs_record==3)[0])
            threes += c
            d = len(np.where(no_jobs_record==4)[0])
            fours += d
            aboves += (no_jobs - a-b-c-d)

    total = iteration*no_jobs*len(spf.m_list)
    #print(total)
    ones_ratio.append(ones/(total))
    #print(ones_ratio)
    twos_ratio.append(twos/(total))
    #print(twos_ratio)
    threes_ratio.append(threes/(total))
    #print(threes_ratio)
    fours_ratio.append(fours/(total))
    #print(fours_ratio)
    above_ratio.append(aboves/(total))
    #print(above_ratio)
    #print('-------')

df_active = DataFrame({'scenarios':scenarios,'ones_ratio':ones_ratio,'twos_ratio':twos_ratio,'threes_ratio':threes_ratio,'fours_ratio':fours_ratio, 'above_ratio':above_ratio})
df_tardy_rate = DataFrame(np.transpose(tardy_rate), columns=scenarios)
df_tardiness = DataFrame(np.transpose(mean_tardiness), columns=scenarios)

Excelwriter = pd.ExcelWriter(sys.path[0]+'\\experiment_result_figure\\RAW_decision_influence.xlsx',engine="xlsxwriter")
dflist = [df_active, df_tardy_rate, df_tardiness]
sheetname = ['active_decision', 'tardy rate', 'avg tardiness']
for i,df in enumerate(dflist):
    df.to_excel(Excelwriter, sheet_name=sheetname[i], index=False)
Excelwriter.save()
