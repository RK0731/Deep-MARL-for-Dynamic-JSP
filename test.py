import random
import copy
import numpy as np
import simpy
import Static_fitness
import Static_spf


class creation:
    def __init__(self, operation_sequence, processing_time, due_date, population_size, generation):
        # get the machine and job number from job's information
        self.m_no = len(set(np.concatenate(operation_sequence)))
        self.j_no = len(operation_sequence)
        # seeds
        self.op_seed = operation_sequence
        self.pt_seed = processing_time
        self.due_date = due_date
        # variables of genetic algorithm
        self.population_size = population_size
        self.generation = generation
        # the seed and CORRECT size of schedule
        self.schedule_seed = []
        for j_idx, ops in enumerate(operation_sequence):
            self.schedule_seed += [j_idx] * len(ops)
        self.schedule_len = len(self.schedule_seed)
        print('Machine number: {} / Job number: {}'.format(self.m_no, self.j_no))
        # the simulation model for fitness evaluation
        self.spf = Static_fitness.shopfloor(operation_sequence, processing_time, due_date)
        # cross over and mutation
        self.mutation_no = int(0.1*self.population_size) # self.population_size * 0.05


    def initialization(self):
        # specify important variables here
        # if this function is not called, all other functions will raise exception
        self.population = []
        self.fitness_value = []
        # create several schedule by certain rule
        #good_start = ['FIFO','SPT','LWKR','MS','WINQ']
        #good_start = ['PTWINQS']
        good_start = []
        for idx,rule in enumerate(good_start):
            # create the environment instance for simulation
            env = simpy.Environment()
            good_start_spf = Static_spf.shopfloor(env, self.op_seed, self.pt_seed, self.due_date, sequencing_rule = rule)
            good_start_spf.simulation()
            print(good_start_spf.job_creator.schedule)
            self.population.append(good_start_spf.job_creator.schedule.copy())
        # fill the rest by random initialization
        for i in range(len(good_start), self.population_size):
            # shuffle first
            random.shuffle(self.schedule_seed)
            # don't forget to use copy
            self.population.append(self.schedule_seed.copy())
        print('initial population:\n',self.population)


    def selection(self):
        #print(self.population, self.fitness_value)
        # sort the fitness_value by ascending order of scores
        self.fitness_value.sort(key = lambda x:x[1])
        curr_best = self.fitness_value[0][1]
        # get the index of chromosomes that needs to be kicked out
        out = self.fitness_value[self.population_size//2:]
        # sort again, by descending order of chromosome index
        out.sort(key = lambda x:x[0], reverse = True)
        # kick out from the end of population
        for t in out:
            self.population.pop(t[0])
        # also, kick half out of fitness record
        self.fitness_value = self.fitness_value[:self.population_size//2]
        # reorder remaining fitness value, to align with the remaining population
        self.fitness_value.sort(key = lambda x:x[0])
        # re-numbering fitness_value
        for idx in range(len(self.fitness_value)):
            self.fitness_value[idx][0] = idx
        #print(self.population, self.fitness_value)
        #print('current best value is:',curr_best)


    def crossover(self):
        # after the selection, half of the population remained as the mating pool
        pool = list(range(self.population_size//2))
        # shuffle the index of the pool
        random.shuffle(pool)
        # get two chromosomes each iteration, so iteration == a quarter of population size
        for i in range(self.population_size//4):
            # randomly draw parents from mating pool
            p1, p2 = pool.pop(), pool.pop()
            # create offspring
            # must use parents' copy to avoid manipulate on parents
            c1, c2 = self.task_crossover(self.population[p1][:], self.population[p2][:])
            # add offsprings to population
            self.population += [c1, c2]
            #print('offspring:', c1, c2)


    def task_crossover(self, x, y):
        # index of jobs that will be directly inherited by offspring
        # whose position in chromosome remain unchanged
        # half of the chromosome will be inherited
        inherit_idx = random.sample(range(self.j_no), self.j_no//2)
        # swap the genes in parents' copy that not directly inherited
        # two pointers exchange
        i, j = 0, 0
        # check parents' all genes, and swap those not directly inherited
        while i < self.schedule_len and j < self.schedule_len:
            if x[i] in inherit_idx and y[j] in inherit_idx:
                i, j = i+1, j+1
            elif x[i] in inherit_idx:
                i += 1
            elif y[j] in inherit_idx:
                j += 1
            else:
                x[i], y[j] = y[j], x[i]
                i, j = i+1, j+1
        return x, y


    def mutation(self):
        # only new offsprings can be mutated
        cand = random.sample(range(self.population_size//2, self.population_size), self.mutation_no)
        for idx in cand:
            # who's gonna be swapped ~
            a, b = random.sample(range(self.schedule_len),2)
            # swap two genes
            self.population[idx][a], self.population[idx][b] = self.population[idx][b], self.population[idx][a]


    def evolution(self):
        # at the beginning, evaluate the fitness of every schedule in the population
        for idx, chromosome in enumerate(self.population):
            score = self.spf.reset_and_execution(chromosome)
            self.fitness_value.append([idx, score])
        self.selection() # kick out half schedules
        self.crossover() # create offsprings
        self.mutation()  # a bit of mutation
        # after the first generation, evaluate only half of the population (offsprings)
        for i in range(1, self.generation):
            # EXECUTE: get the fitness score for the latter half of the population (offsprings)
            for idx, chromosome in enumerate(self.population[self.population_size//2:], start = self.population_size//2):
                score = self.spf.reset_and_execution(chromosome)
                self.fitness_value.append([idx, score])
            #print(self.population, self.fitness_value)
            self.selection() # kick out half schedules
            self.crossover() # create offsprings
            self.mutation()  # a bit of mutation
        # terminate the evolution and output the best schedule
        self.output()


    def output(self):
        # evaluate the final population
        for idx, chromosome in enumerate(self.population):
            score = self.spf.reset_and_execution(chromosome)
            self.fitness_value.append((idx, score))
        # sort by fitness score, asceding order of scores
        self.fitness_value.sort(key = lambda x:x[1])
        # the first one is the best schedule's (index, score)
        best = self.fitness_value[0]
        #print(self.population)
        #print(self.fitness_value)
        print('\n','-'*100,'\n','EVOLUTION COMPLETED\n','-'*100,'\n')
        print('Best schedule is:\n',self.population[best[0]])
        # get the transformed schedule of the best chromosome
        # which is the actual execution order of all operations
        self.spf.reset_and_execution(self.population[best[0]])
        transformed_schedule = [item[0] for item in self.spf.record]
        print('Lowest tardiness is:\n',self.fitness_value[0][1])
        print('Tardy job no.: ', self.spf.tardy_no, self.spf.tardy_record)
        print('Tansformed schedule is:\n', transformed_schedule)
        # return the transformed schedule
        return transformed_schedule