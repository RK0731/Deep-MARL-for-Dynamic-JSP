import simpy
import sys
sys.path #sometimes need this to refresh the path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

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

        # STEP 2: create instances of machines
        for i in range(m_no):
            expr1 = '''self.m_{} = Machine.machine(env, {}, print = 0)'''.format(i,i) # create machines
            exec(expr1)
            expr2 = '''self.m_list.append(self.m_{})'''.format(i) # add to machine list
            exec(expr2)

        # STEP 3: initialize the initial jobs, distribute jobs to workcenters
        # env, span, machine_list, workcenter_list, number_of_jobs, pt_range, due_tightness, E_utliz, print
        self.job_creator = Event_job_creation.creation\
        (self.env, self.span, self.m_list, [1,50], 3, 0.9, print = 0)
        self.job_creator.initial_output()

        # STEP 4: initialize all machines
        for i in range(m_no):
            expr3 = '''self.m_{}.initialization(self.m_list,self.job_creator)'''.format(i) # initialize all machines
            exec(expr3)

        # STEP 5: check if need to reset sequencing rule
        if 'sequencing_rule' in kwargs:
            print("Taking over: machines use {} sequencing rule".format(kwargs['sequencing_rule']))
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

# create the environment instance for simulation
env = simpy.Environment()
# create the shop floor instance
# the command of startig the simulation is included in shopfllor instance, run till there's no more events
span = 500
spf = shopfloor(env, span, 10, sequencing_rule="FIFO")
spf.simulation()

'''
generate the figures
'''
# create the figure instance
fg1 = plt.figure(figsize=(10,7))
# concatenate the produciton records from all machines
breakdown_record = []
last_output = []
for Asset_machine in spf.m_list:
    breakdown_record += Asset_machine.breakdown_record
# upper half of fg1
gantt_chart = fg1.add_subplot(2,1,1)
bar_pos = np.stack([np.arange(1, spf.m_no + 1)-0.25, np.ones(spf.m_no)/2], axis=1) # vertical position of bars
#print(bar_pos)
GC_yticks_pos = np.arange(1, spf.m_no + 1) # vertical position of tick labels
col_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
# plot the operations
p_record = spf.job_creator.production_record
# each item contains records of all operations of a job
for item in p_record:
    #print(p_record[item])
    # iterate all sub-items, operations after operations
    for k in range(len(p_record[item][0])):
        #print(p_record[item][0][k], p_record[item][1][k], bar_pos[p_record[item][1][k]])
        gantt_chart.broken_barh([p_record[item][0][k]], bar_pos[p_record[item][1][k]], color=col_list[item%10], edgecolor='w')
        # put the index of job on gantt chart
        gantt_chart.text(p_record[item][0][k][0]+p_record[item][0][k][1]/2, p_record[item][1][k]+0.95, item, fontsize=8, ha='center', va='center', color='w')
# find the output time of last job, determine the range of plot
#print(spf.job_creator.production_record[spf.job_creator.no_jobs-1])
last_output = np.max(spf.job_creator.production_record[spf.job_creator.index_jobs-1][3])
plot_range = np.ceil(last_output/10)*10
#print(last_output,plot_range)
# plot machine breakdowns
for item in breakdown_record:
    gantt_chart.broken_barh([item[0]], bar_pos[item[1]], color='w', edgecolor='k', hatch='//')
    gantt_chart.text(item[0][0]+item[0][1]/2, item[1]+0.95, 'BKD.', fontsize=8, ha='center', va='center', color='r')
# labels, ticks and grids
gantt_chart.set_xlabel('Time of simulation')
gantt_chart.set_ylabel('Machines')
gantt_chart.set_title('The production record of machines (Gantt Chart)')
gantt_chart.set_yticks(GC_yticks_pos)
gantt_chart.set_yticklabels([machine.m_idx for machine in spf.m_list])
# set grid and set grid behind bars
fg1_major_ticks = np.arange(0, plot_range+1, 10)
fg1_minor_ticks = np.arange(0, plot_range+1, 1)
# Major ticks every 20, minor ticks every 5
gantt_chart.set_xticks(fg1_major_ticks)
gantt_chart.set_xticks(fg1_minor_ticks, minor=True)
# different settings for the grids:
gantt_chart.grid(which='major', alpha=1)
gantt_chart.grid(which='minor', alpha=0.2, linestyle='--')
gantt_chart.set_axisbelow(True)
# limit
gantt_chart.set_xlim(0, plot_range)
gantt_chart.set_ylim(0)
#gantt_chart.legend()

# lower half of fg1, retrive the data
tard_record = fg1.add_subplot(2,1,2)
output_time, cumulative_tard, avg_tard, tard_max, tard_rate = spf.job_creator.tardiness_output()
# labels and title
color = "r"
tard_record.set_xlabel('Time of simulation')
tard_record.set_ylabel('Cumulative Tardiness', color = color)
tard_record.set_title('Tardiness of jobs')
tard_record.plot(output_time, cumulative_tard, label='Cumulative Tardiness',color=color)
# Major ticks every 20, minor ticks every 5
tard_record.set_xticks(fg1_major_ticks)
# different settings for the grids:
tard_record.grid(axis='x')
tard_record.set_axisbelow(True)
# set secondary y axis
color = "b"
sec_tard_record = tard_record.twinx()
sec_tard_record.set_ylabel('Tardiness per piece', color = color)
sec_tard_record.plot(output_time, avg_tard, label='Tardiness/Pics',linestyle='--', color=color)
# limit
tard_record.set_xlim(0,plot_range)
tard_record.set_ylim(0)
sec_tard_record.set_ylim(0)


plt.show()

'''
def transform3_action_NN(axis):
    return axis * spf.brain.action_NN_training_interval + spf.brain.begin
# inverse means how to transform the axis back to the original axis
def inverse3_action_NN(axis):
    return (axis - spf.brain.begin) / spf.brain.action_NN_training_interval
axe3=fig1.add_subplot(1,2,2)
iteration=np.arange(len(spf.brain.loss_record))
axe3.plot(iteration, spf.brain.loss_record)
vertical_lines = inverse3_action_NN(np.array(spf.brain.target_NN_update_time_record,dtype=float))
#axe3.vlines(vertical_lines,0,0.01, color='r', label='target_NN weight update')
axe3.set_xlabel('Number of iteration')
axe3.set_ylabel('Loss')
axe3.set_title('The loss of NN')
axe3.grid()
axe3.set_xlim(0,len(spf.brain.loss_record))
axe3.set_ylim(0)
secax = axe3.secondary_xaxis('top', functions=(transform3_action_NN, inverse3_action_NN))
secax.set_xlabel('Time') # double x axis,  on the top

# the following are code for the second figure

fig2 = plt.figure()

gantt_chart=fig2.add_subplot(2,1,1)
bar_pos = np.stack([np.arange(1,5)-0.5,np.ones(4)/2],axis=1) # vertical position of bars
GC_yticks_pos = np.arange(4)+0.75 # vertical position of tick labels
gantt_chart.broken_barh(spf.m_A.production_record,bar_pos[3],label='A',color='r')
gantt_chart.broken_barh(spf.m_B.production_record,bar_pos[2],label='B',color='g')
gantt_chart.broken_barh(spf.m_C.production_record,bar_pos[1],label='C',color='b')
gantt_chart.broken_barh(spf.m_D.production_record,bar_pos[0],label='D',color='y')
gantt_chart.set_xlabel('Time')
gantt_chart.set_ylabel('Machines')
gantt_chart.set_title('The production periods of machines (Gnatt Chart)')
gantt_chart.set_yticks(GC_yticks_pos)
gantt_chart.set_yticklabels(['A', 'B', 'C','D'])
#the indicator of protection period and pre-training period
gantt_chart.axvline(spf.brain.protection_period, ls='--', color='r')
gantt_chart.text(spf.brain.protection_period, 0.2, "(protection)", fontsize=10, verticalalignment="center")
gantt_chart.axvline(spf.brain.pre_training_period, ls='--', color='b')
gantt_chart.axvline(span)
gantt_chart.text(spf.brain.pre_training_period, 0.2, "(hand over to NN)", fontsize=10, verticalalignment="center")
gantt_chart.grid()
gantt_chart.set_xlim(0,span*1.08)
gantt_chart.set_ylim(0)
gantt_chart.legend()

'''
