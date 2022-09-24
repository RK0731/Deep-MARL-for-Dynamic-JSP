import simpy
import time
import copy
import numpy as np
from tabulate import tabulate
from Static_fitness import fitness_test
from Static_spf import shopfloor



with open('test.txt') as f:
    lines = f.readlines()
operation_sequence = eval(lines[0])
processing_time = eval(lines[1])
due_date = eval(lines[2])

print(operation_sequence)
print(processing_time)
print(due_date)
print('-'*50)
'''
env = simpy.Environment()
spf = shopfloor(env, operation_sequence, processing_time, due_date, sequencing_rule = 'PTWINQS')
spf.simulation()
print(spf.job_creator.schedule)
output_time, cumulative_tard, tard_mean, tard_max, tard_rate = spf.job_creator.tardiness_output()
print(cumulative_tard[-1])
'''

fitness_test([2, 1, 3, 0, 0, 2, 3, 3, 1, 4, 0, 1, 3, 0, 4, 2, 3, 0, 1, 4, 2, 1, 2, 4, 4], 0)
