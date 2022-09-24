from Static_spf import shopfloor
import simpy
import sys
import Static_genetic_algorithm
import pandas as pd
import time

path = sys.path
with open('test.txt') as f:
    lines = f.readlines()
operation_sequence = eval(lines[0])
processing_time = eval(lines[1])
due_date = eval(lines[2])

#benchmark = ['SPT','MS','PTWINQS']
benchmark = ['FIFO','PTWINQS']

sum_record = []
max_record = []
rate_record = []
no_record = []

''' benchmark priority rules '''
for idx,rule in enumerate(benchmark):
    # create the environment instance for simulation
    env = simpy.Environment()
    spf = shopfloor(env, operation_sequence, processing_time, due_date, sequencing_rule = rule)
    spf.simulation()
    output_time, cumulative_tard, tard_mean, tard_max, tard_rate = spf.job_creator.tardiness_output()
    cnt = 0
    if cumulative_tard[0]>0:
        cnt+=1
    for idx in range(1,len(cumulative_tard)):
        if cumulative_tard[idx] != cumulative_tard[idx-1]:
            cnt += 1
    no_record.append(cnt)
    sum_record.append(cumulative_tard[-1])
    max_record.append(tard_max)
    rate_record.append(tard_rate)
    print(spf.job_creator.schedule)


''' deep MARL / DRL approach '''
env = simpy.Environment()
spf = shopfloor(env, operation_sequence, processing_time, due_date, MR = True)
spf.simulation()
output_time, cumulative_tard, tard_mean, tard_max, tard_rate = spf.job_creator.tardiness_output()
cnt = 0
if cumulative_tard[0]>0:
    cnt+=1
for idx in range(1,len(cumulative_tard)):
    if cumulative_tard[idx] != cumulative_tard[idx-1]:
        cnt += 1
no_record.append(cnt)
sum_record.append(cumulative_tard[-1])
max_record.append(tard_max)
rate_record.append(tard_rate)
print('deep MARL schedule: ', spf.job_creator.schedule)

print('Cumulative tardiness')
print(sum_record)
print('No. of tardy jobs')
print(no_record)

''' Genetic Algorithm '''
population_size = 100
generation = 50
GA = Static_genetic_algorithm.creation(operation_sequence, processing_time, due_date, population_size, generation)
start = time.time()
GA.initialization()
GA.evolution()
