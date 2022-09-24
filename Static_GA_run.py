import Static_genetic_algorithm
import pandas as pd
import time


with open('test.txt','r') as f:
    lines = f.readlines()
operation_sequence = eval(lines[0])
processing_time = eval(lines[1])
due_date = eval(lines[2])
population_size = 100
generation = 50


GA = Static_genetic_algorithm.creation(operation_sequence, processing_time, due_date, population_size, generation)

start = time.time()

GA.initialization()
GA.evolution()

print('CPU time for GA: ', time.time()-start)
