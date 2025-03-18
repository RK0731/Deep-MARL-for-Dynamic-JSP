from src.static_experiment.static_genetic_algorithm import GA
import pandas as pd
import time


with open('test.txt','r') as f:
    lines = f.readlines()
operation_sequence = eval(lines[0])
processing_time = eval(lines[1])
due_date = eval(lines[2])
population_size = 100
generation = 50


ga_agent = GA(operation_sequence, processing_time, due_date, population_size, generation)

start = time.time()

ga_agent.initialization()
ga_agent.evolution()

print('CPU time for GA: ', time.time()-start)
