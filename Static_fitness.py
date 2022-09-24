import simpy
import time
import copy
import pandas as pd
from tabulate import tabulate
from pandas import DataFrame
import numpy as np
import Static_gantt_chart


'''
Simplified simulation model based on SimPy for fastest fitness evaluation
'''
class shopfloor:
    def __init__(self, operation_sequence, processing_time, due_date, **kwargs):
        # get the machine and job number from job's information
        self.m_no = len(set(np.concatenate(operation_sequence)))
        self.j_no = len(operation_sequence)
        # seed will remain unchanged during the whole fitness evaluation
        self.op_seed = operation_sequence
        self.pt_seed = processing_time
        self.due_date = due_date
        # to store tuples storing information of each operation
        # check input
        if self.j_no != len(operation_sequence):
            print("ERROR: mismatch number of jobs and operation_sequence !")
            raise Exception
        if np.array(operation_sequence, dtype=object).shape != np.array(processing_time, dtype=object).shape:
            print("ERROR: mismatch shape of op_sqc and pt !")
            raise Exception


    def check_schedule(self, schedule, op_sqc):
        # check if the schedule contains all operations
        # for problem where jobs have hetero-length operation sequence
        if len(np.concatenate(op_sqc)) != len(schedule):
            print('ERROR: mismatch schedule and operation sequence')
            raise Exception


    def reset_and_execution(self, schedule):
        # RESET: create an new environment and reset cumulative tardiness and record
        self.env, self.tardiness, self.tardy_no, self.tardy_record, self.record = simpy.Environment(), 0, 0, [], []
        # MUST use deepcopy, otherwise pop the element from list in list would change the source list
        self.schedule = copy.deepcopy(schedule)
        # empty operation sequence and processing time
        self.op_sqc, self.pt = [], []
        # MUST use deepcopy, create the stacks for re-ordering
        stack_op_sqc, stack_pt = copy.deepcopy(self.op_seed), copy.deepcopy(self.pt_seed)
        # STACK: remaining operation for jobs
        self.rem_op_sqc, self.rem_pt = copy.deepcopy(self.op_seed), copy.deepcopy(self.pt_seed)
        # RESET: re-order the operations and processing time to align with the schedule
        for j_idx in self.schedule:
            self.op_sqc.append(stack_op_sqc[j_idx].pop(0))
            self.pt.append(stack_pt[j_idx].pop(0))
        # at this point, we have schedule, op_sqc, and pt
        #print(self.schedule)
        #print(self.op_sqc)
        #print(self.pt)
        # RESET: create machines as resources
        self.m_list = [] # clear all machines in last round
        for i in range(self.m_no):
            self.m_list.append(simpy.Resource(self.env, capacity=1))
        # RESET: mark all machines as available to be processed at the beginning
        self.m_idle = [1]*self.m_no
        # RESET: mark all jobs as available as well
        self.j_available = [1]*self.j_no
        # EXECUTE: perform all operations by the schedule
        self.initialization()
        self.env.run()
        return self.tardiness


    def initialization(self):
        cnt, kick = 0, [] # machine count, and operations that gonna be kicked out of op_sqc
        # go through the schedule
        for i, m_idx in enumerate(self.op_sqc):
            # processing time and job's index that correspond to that operation
            pt, j_idx = self.pt[i], self.schedule[i]
            # CRITERION: if machine is idle and job's next operation needs that machine
            if self.m_idle[m_idx] and self.rem_op_sqc[j_idx][0] == m_idx:
                self.env.process(self.operation(m_idx, pt, j_idx))
                cnt += 1
                kick.insert(0,i)
                self.m_idle[m_idx] = 0
                self.j_available[j_idx] = 0
            # terminate the for loop if all machines are occupied
            if cnt == self.m_no:
                break
        # after the initialization, kick out elements from stacks
        for idx in kick:
            self.op_sqc.pop(idx)
            self.pt.pop(idx)
            self.schedule.pop(idx)


    def operation(self, m_idx, pt, j_idx):
        # CRITICAL PRECEDENCE: yield dummy time to build the precedence of operation
        # CRITICAL PRECEDENCE: otherwise jobs would seize all resources at time 0
        yield self.env.timeout(0)
        # LOCK: occupy the machine for the operation of job
        req = self.m_list[m_idx].request()
        yield req
        self.record.append((j_idx, self.env.now, pt, m_idx))
        #print('job {} start operation at machine {} at time {}'.format(j_idx, m_idx, self.env.now))
        yield self.env.timeout(pt)
        # UNLOCK: release the machine after operation
        #print('job {} left machine {} at time {}'.format(j_idx, m_idx, self.env.now))
        self.m_list[m_idx].release(req)
        # mark both the machine and job as available
        self.m_idle[m_idx], self.j_available[j_idx] = 1, 1
        # after the operation, revise the stack of remaining operation of job
        # then the job can be selected for its next operation
        self.rem_op_sqc[j_idx].pop(0)
        # CRITICAL PRECEDENCE: avoid seizing resources by completion time (instead of schedule)
        yield self.env.timeout(0)
        # NEXT: check available machines in system and queuing jobs
        self.next_operation(j_idx)


    def next_operation(self, completed_j_idx):
        # JOB CHECK: check the status of job
        # if the job is completed, add its tardiness to cum-sum
        if not self.rem_op_sqc[completed_j_idx]:
            tard = max(0, self.env.now - self.due_date[completed_j_idx])
            self.tardiness += tard
            self.tardy_record.append((completed_j_idx, self.env.now, self.due_date[completed_j_idx]))
            if tard:
                self.tardy_no += 1
            self.j_available[completed_j_idx] = 0
        # MACHINE CHECK: check is there any idle machines that can be used
        m_can_use = [m_idx for m_idx in range(self.m_no) if self.m_idle[m_idx] == 1]
        # machine count, stop criterion, kick out set
        cnt, stop, kick = 0, len(m_can_use), []
        #print(self.env.now, m_can_use)
        # SCHEDULE CHECK: go through remaining operations
        for i, m_idx in enumerate(self.op_sqc):
            # processing time and job's index that correspond to that operation
            pt, j_idx = self.pt[i], self.schedule[i]
            # CRITERION: if machine is idle and job's next operation needs that machine
            if m_idx in m_can_use and self.m_idle[m_idx] and self.rem_op_sqc[j_idx][0] == m_idx:
                #print('use machine {} to process job {}'.format(m_idx, j_idx))
                self.env.process(self.operation(m_idx, pt, j_idx))
                cnt += 1
                kick.insert(0,i)
                self.m_idle[m_idx] = 0
                self.j_available[j_idx] = 0
            # terminate the for loop if all machines are occupied
            if cnt == stop:
                break
        # after the allocation of job/machine, kick out elements from stacks
        for idx in kick:
            self.op_sqc.pop(idx)
            self.pt.pop(idx)
            self.schedule.pop(idx)


'''
Test
'''
class fitness_test:
    def __init__(self, schedule, plot):
        with open('test.txt') as f:
            lines = f.readlines()
        operation_sequence = eval(lines[0])
        processing_time = eval(lines[1])
        due_date = eval(lines[2])
        spf = shopfloor(operation_sequence, processing_time, due_date)
        spf.check_schedule(schedule, operation_sequence)
        for i in range(1):
            score = spf.reset_and_execution(schedule)
            print(spf.record)
            print("Cumulative tardiness: ",spf.tardiness)
            print('Tardy job no.: ', spf.tardy_no)
        if plot:
            Static_gantt_chart.draw(spf.record, spf.m_no, spf.due_date)
