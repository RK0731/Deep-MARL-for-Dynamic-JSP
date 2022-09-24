import simpy
import sys
sys.path
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from tabulate import tabulate
import Rule_sequencing


class machine:
    def __init__(self, env, index, *args, **kwargs):
        # initialize the environment of simulation
        self.env = env
        self.m_idx = index

        # each machine will have an independent storage for each type of job information
        # initialize all job-related information storage as empty lists
        self.queue = []
        self.sequence_list = [] # sequence of all queuing jobs
        self.remaining_pt_list = [] # processing time of remaining operations (include current operation)
        self.due_list = [] # due for each job
        self.arrival_time_list = [] # time that job join the queue
        self.waited_time = [] # time that job stayed in the queue
        self.slack_upon_arrival = [] # slack record of queuing jobs upon their arrival
        self.breakdown_record = []
        self.no_jobs_record = []
        self.reward_record = [[],[]] # record the trend of collecting rewards

        # important time points
        self.decision_point = 0 # time point of decision-making
        self.release_time = 0 # time when machinhe release from current operation, doesn't necessarily coincide decision-making point
        self.cumulative_run_time = 0 # track the utilization rate
        self.restart_time = 0 # this is the time that machine needs to recover from breakdown

        # Initialize the possible events during production
        self.sufficient_stock = self.env.event() # queue is 0 or not
        self.working_event = self.env.event() # working condition in shut down and breakdowns

        # Initialize the events'states
        self.working_event.succeed()

        # performance measurement
        # use exponential moving average to measure slack and tardiness
        self.EMA_slack_change = 0
        self.EMA_realized_tardiness = 0
        self.EMA_alpha = 0.1
        # set the sequencing rule before start of simulation
        if 'rule' in kwargs:
            order = "self.job_sequencing = sequencing." + kwargs['rule']
            try:
                exec(order)
                print("machine {} uses {} sequencing rule".format(self.m_idx, kwargs['rule']))
            except:
                print("Rule assigned to machine {} is invalid !".format(self.m_idx))
                raise Exception
        else:
            # default sequencing rule is FIFO
            self.job_sequencing = Rule_sequencing.FIFO
        # print the operations or not
        self.print = 0 # don't print by default
        if 'print' in kwargs:
            self.print = kwargs['print']
        # record extra data for learning, initially not activated, can be activated by brains
        self.sequencing_learning_event = self.env.event()
        self.routing_learning_event = self.env.event()


    '''
    1. downwards are functions that perform the simulation
       including production, starvation and breakdown
    '''


    # this function should be called after __init__ to avoid deadlock
    # after the creation of all machines and initial jobs
    # the initial jobs are allocated through job_creation module
    def initialization(self, machine_list, job_creator):
        # knowing other machines, workcenters, and the job creator
        # so the machine agent can manipulate other agents'variables
        self.m_list = machine_list
        self.m_no = len(self.m_list)
        self.no_ops = self.m_no
        self.job_creator = job_creator
        # initial information
        if self.print:
            print('Initial %s jobs at machine %s are:'%(len(self.queue), self.m_idx))
            job_info = [[self.queue[i],self.sequence_list[i], self.remaining_pt_list[i], self.slack_upon_arrival[i], self.due_list[i]] for i in range(len(self.queue))]
            print(tabulate(job_info, headers=['idx.','sqc.','rem.pt','slack','due']))
            print('************************************')
        self.state_update_after_initialization()
        self.state_update_global_info_progression()
        self.env.process(self.production())

    # The main function, simulates the production
    def production(self):
        if not self.sufficient_stock.triggered:
            yield self.env.process(self.starvation())
        # the loop that will run till the ned of simulation
        while True:
            '''STEP 0: update queueing job information, record decision-making point'''
            self.state_update_all()
            self.decision_point = self.env.now
            self.no_jobs_record.append(len(self.queue))
            '''STEP 1: sequencing decision-making'''
            # + np.any(self.job_creator.next_machine_list==self.m_idx)*1
            if len(self.queue) - 1: # if more than one candidate job
                self.position, strategic_idleness_bool, strategic_idleness_time = self.job_sequencing(self.sequencing_data_generation())
                if strategic_idleness_bool: # if the decision is to wait for the arriving job
                    if self.print:
                        print("Machines %s strategic idleness at time %s for %s units, position: %s "%(self.m_idx, self.env.now, strategic_idleness_time, self.position))
                    yield self.env.timeout(strategic_idleness_time) # allows strategic idleness
                    self.decision_point = self.env.now # refresh the decision point
                    self.state_update_all() # and update the information again after the idleness
                #print(self.queue, self.position)
                self.job_idx = self.queue[self.position] # the index of selected job (the job that just arrived)
                #self.reward_preparation()
                if self.print:
                    print("Sequencing: Machine %s choose job %s at time %s"%(self.m_idx,self.job_idx,self.env.now))
            else: # if there's only one job can be picked
                self.position = 0
                self.job_idx = self.queue[self.position]
                if self.print:
                    print("One job: Machine %s choose job %s at time %s"%(self.m_idx,self.job_idx,self.env.now))
            '''STEP 2: after the job to be processed is determined, retrive its information'''
            pt = self.remaining_pt_list[self.position][0] # processing time of the selected job on machine
            wait = self.env.now - self.arrival_time_list[self.position] # time that job waited before being selected
            # after determined the next job to be processed, update a bunch of data
            self.state_update_global_info_progression()
            self.state_update_global_info_anticipation(pt)
            self.record_production(pt, wait) # record these information
            self.job_creator.schedule.append(self.job_idx)
            self.job_creator.op_record.append((self.job_idx, self.env.now, pt, self.m_idx))
            '''STEP 3: production process (transition to next state)'''
            yield self.env.timeout(pt)
            self.cumulative_run_time += pt
            #print("completion: Job %s leave machine %s at time %s"%(self.queue[self.position],self.m_idx,self.env.now))
            # transfer job to next workcenter or delete it, and update information
            self.route_after_operation()
            '''STEP 4: breakdown and starvation (optional), then go to next loop'''
            if not self.working_event.triggered:
                yield self.env.process(self.breakdown())
            # check the queue/stock level, if none, starvation begines
            if not len(self.queue):
                yield self.env.process(self.starvation())

    def starvation(self):
        #print('STARVATION *BEGIN*: machine %s at time %s' %(self.m_idx, self.env.now))
        # set the self.sufficient_stock event to untriggered
        self.sufficient_stock = self.env.event()
        # proceed only if the sufficient_stock event is triggered by new job arrival
        yield self.sufficient_stock
        # examine whether the scheduled shutdown is triggered
        if not self.working_event.triggered:
            yield self.env.process(self.breakdown())
        #print('STARVATION *END*: machine %s at time: %s'%(self.m_idx, self.env.now))

    def breakdown(self):
        print('********', self.m_idx, "breakdown at time", self.env.now, '********')
        start = self.env.now
        # simply update the available time of that machines
        self.available_time = self.restart_time + self.cumulative_pt
        # suspend the production here, untill the working_event is triggered
        yield self.working_event
        self.breakdown_record.append([(start, self.env.now-start), self.m_idx])
        print('********', self.m_idx, 'brekdown ended, restart production at time', self.env.now, '********')


    '''
    2. downwards are functions the called before and after each operation
       to maintain some record, and transit the finished job to next workcenter or out of system
    '''


    # update lots information that will be used for calculating the rewards
    def reward_preparation(self):
        # number of jobs that to be sequenced, and their ttd and slack
        self.waiting_jobs = len(self.queue)
        time_till_due = np.array(self.due_list) - self.env.now
        self.before_op_ttd = time_till_due
        self.before_op_ttd_chosen = self.before_op_ttd[self.position]
        self.before_op_ttd_loser = np.delete(self.before_op_ttd, self.position)
        tardy_jobs = len(time_till_due[time_till_due<0])
        #self.before_op_realized_tard_rate =tardy_jobs/len(self.queue)
        #print('before realized tard rate: ', self.before_op_realized_tard_rate)
        initial_slack = self.slack_upon_arrival.copy()
        self.before_op_remaining_pt = self.remaining_job_pt + self.current_pt
        self.before_op_remaining_pt_chosen = self.before_op_remaining_pt[self.position]
        self.before_op_remaining_pt_loser = np.delete(self.before_op_remaining_pt, self.position)
        current_slack = time_till_due - self.before_op_remaining_pt
        exp_tardy_jobs = len(current_slack[current_slack<0])
        # get information of all jobs before operation
        self.before_op_exp_tard = current_slack[current_slack<0]
        self.before_op_sum_exp_tard = self.before_op_exp_tard.sum()
        self.before_op_slack = current_slack
        self.before_op_sum_slack = self.before_op_slack.sum()
        # calculate the critical level  of all queuing jobs
        self.critical_level = 1 - current_slack / 100
        self.critical_level_chosen  = self.critical_level[self.position]
        #print(current_slack, self.critical_level,self.critical_level_chosen)
        # get the information of the selected job
        self.pt_chosen = self.current_pt[self.position]
        self.initial_slack_chosen = initial_slack[self.position]
        self.before_op_slack_chosen = current_slack[self.position]
        self.before_op_exp_tard_chosen = min(0,self.before_op_slack_chosen)
        self.before_op_winq_chosen = self.winq[self.position]
        # get the information of jobs that haven't been selected (loser)
        self.before_op_slack_loser = np.delete(current_slack, self.position) # those haven't been selected
        self.critical_level_loser = np.delete(self.critical_level, self.position)
        self.before_op_sum_exp_tard_loser = self.before_op_slack_loser[self.before_op_slack_loser<0].sum()
        self.before_op_sum_slack_loser = self.before_op_slack_loser.sum()
        self.before_op_winq_loser = np.delete(self.winq, self.position)
        #print('before',self.m_idx,self.env.now,slack,slack_loser,self.before_op_exp_tard,self.current_pt,self.position)
        #self.before_op_avg_slack = slack.sum()/len(self.queue)
        #self.before_op_expected_tard_rate = exp_tardy_jobs/len(self.queue)
        #print('before expected tard rate: ', self.before_op_expected_tard_rate)

    # transfer unfinished job to next workcenter, or delete finished job from record
    # and update the data of queuing jobs, EMA_tardiness etc.
    def route_after_operation(self):
        # check if this is the last operation of job
        # if the sequence is not empty, any value > 0 is True
        if len(self.sequence_list[self.position]):
            if self.print:
                print('OPERATION: Job %s output from machine %s at time %s'%(self.queue[self.position], self.m_idx, self.env.now))
            next_machine = self.sequence_list[self.position][0]
            # add the information of this job to next machine's storage
            self.m_list[next_machine].queue.append(self.queue.pop(self.position))
            self.m_list[next_machine].sequence_list.append(np.delete(self.sequence_list.pop(self.position),0))
            # update the information for completed job before route to next machine
            remaining_ptl = np.delete(self.remaining_pt_list.pop(self.position),0)
            next_pt = remaining_ptl[0] # pt of NEXT operation
            current_slack = self.due_list[self.position] - self.env.now - remaining_ptl.sum()
            original_slack = self.slack_upon_arrival.pop(self.position)
            # add the information of this job to next machine's storage
            self.m_list[next_machine].remaining_pt_list.append(remaining_ptl)
            self.m_list[next_machine].due_list.append(self.due_list.pop(self.position))
            self.m_list[next_machine].slack_upon_arrival.append(current_slack)
            self.m_list[next_machine].arrival_time_list.append(self.m_list[next_machine].release_time)
            del self.arrival_time_list[self.position]
            # add update the next machine's information
            self.m_list[next_machine].state_update_after_job_arrival(next_pt)
            # calculate slack gain/loss, and record it
            self.slack_change = current_slack - original_slack
            self.record_slack_tardiness(original_slack)
            # and activate the dispatching of next work center
            try:
                self.m_list[next_machine].sufficient_stock.succeed()
            except:
                pass
            # update all information needed for decision-making
            self.state_update_all()
            self.state_update_global_info_after_operation()
            # check if sequencing learning mode is on, and queue is not 0
            if self.sequencing_learning_event.triggered:
                self.record_state()
        # if this is the last process, then simply delete job information
        else:
            if self.print:
                print('**FINISHED: Job %s from machine %s at time %s'%(self.queue[self.position], self.m_idx, self.env.now))
            # calculate tardiness of job, and update EMA_realized_tardiness
            self.tardiness = np.max([0, self.env.now - self.due_list[self.position]])
            #print("realized tardiness is:", tardiness)
            self.EMA_realized_tardiness += self.EMA_alpha * (self.tardiness - self.EMA_realized_tardiness)
            #print(self.m_idx,self.EMA_realized_tardiness)
            # delete the information of this job
            del self.queue[self.position]
            del self.sequence_list[self.position]
            del self.remaining_pt_list[self.position]
            # get old and current_slack time of the job
            current_slack = self.due_list[self.position] - self.env.now # there's no more operations for this job
            del self.due_list[self.position]
            original_slack = self.slack_upon_arrival.pop(self.position)
            del self.arrival_time_list[self.position]
            # kick the job out of system
            self.job_creator.record_job_departure()
            #print(self.job_creator.in_system_job_no)
            # calculate slack gain/loss
            self.slack_change = current_slack - original_slack
            # record the slack change
            self.record_slack_tardiness(original_slack, self.tardiness)
            #print("original_slack: %s / current_slack: %s"%(original_slack, current_slack))
            # update all information needed for decision-making
            self.state_update_all()
            self.state_update_global_info_after_operation()
            # check if sequencing learning mode is on, and queue is not 0
            # if yes, since the job is finished and tardiness is realized, construct complete experience
            if self.sequencing_learning_event.triggered:
                self.record_state()
                self.reward_function() # job's actual tardiness is realized, thus reward can be calculated


    '''
    3. downwards are functions that related to information update and exchange
       especially the information that will be used by other agents on shop floor
    '''


    def record_production(self, pt, wait): # called before production
        # add the details of operation to job_creator's repository
        self.job_creator.production_record[self.job_idx][0].append((self.env.now, pt))
        self.job_creator.production_record[self.job_idx][1].append(self.m_idx)
        self.job_creator.production_record[self.job_idx][2].append(wait)

    def record_slack_tardiness(self, original_slack, *args): # called after production
        self.job_creator.production_record[self.job_idx][3].append(original_slack)
        if len(args): # if this is the last operation (*arg is not null)
            self.job_creator.production_record[self.job_idx].append((self.env.now,args[0]))

    def update_reward_record(self, r_t):
        self.reward_record[0].appned(self.env.now)
        self.reward_record[1].append(r_t)

    # call this function after the initialization, used for once
    def state_update_after_initialization(self):
        self.current_pt = np.array([x[0] for x in self.remaining_pt_list])
        self.cumulative_pt = self.current_pt.sum()
        self.available_time = self.env.now + self.cumulative_pt
        self.remaining_job_pt = np.array([x.sum() for x in self.remaining_pt_list])
        self.remaining_no_op = np.array([len(x) for x in self.remaining_pt_list])
        self.next_pt = np.array([x[1] if len(x)-1 else 0 for x in self.remaining_pt_list])
        self.completion_rate = np.array([(self.no_ops-len(x)-1)/self.no_ops for x in self.remaining_pt_list])
        self.que_size = len(self.queue)
        self.time_till_due = np.array(self.due_list) - self.env.now
        self.slack = self.time_till_due - self.remaining_job_pt
        self.waited_time = np.array([0 for x in self.sequence_list])
        # dummy WINQ and avlm
        self.winq = np.array([0 for x in self.sequence_list])
        self.avlm = np.array([0 for x in self.sequence_list])

    # call this function before sequencing decision / after operation / after breakdown and starvation
    def state_update_all(self):
        # processing time of current process of each queuing job
        self.current_pt = np.array([x[0] for x in self.remaining_pt_list])
        # cumultive processing time of all queuing jobs on this machine
        self.cumulative_pt = self.current_pt.sum()
        # the time the machine will be available (become idle or breakdown ends)
        self.available_time = self.env.now + self.cumulative_pt
        # expected cumulative processing time (worst possible) of all unfinished processes for each queuing job
        self.remaining_job_pt = np.array([x.sum() for x in self.remaining_pt_list])
        self.remaining_no_op = np.array([len(x) for x in self.remaining_pt_list])
        self.next_pt = np.array([x[1] if len(x)-1 else 0 for x in self.remaining_pt_list])
        # the completion rate of all queuing jobs
        self.completion_rate = np.array([(self.no_ops-len(x)-1)/self.no_ops for x in self.remaining_pt_list])
        # number of queuing jobs
        self.que_size = len(self.queue)
        # time till due and slack time of jobs
        self.time_till_due = np.array(self.due_list) - self.env.now
        self.slack = self.time_till_due - self.remaining_job_pt
        # time that job spent in the queue
        self.waited_time = self.env.now - np.array(self.arrival_time_list)
        # WINQ and available time of all machines
        self.winq = np.array([self.m_list[x[0]].cumulative_pt if len(x) else 0 for x in self.sequence_list])
        self.avlm = np.array([max(self.m_list[x[0]].available_time - self.env.now, 0) if len(x) else 0 for x in self.sequence_list])
        #print(self.winq, self.avlm)

    # available time is a bit tricky, jobs may come when the operation is ongoing
    # or when the machine is starving (availble time is earlier than now)
    def state_update_after_job_arrival(self, increased_available_time):
        self.current_pt = np.array([x[0] for x in self.remaining_pt_list])
        self.cumulative_pt = self.current_pt.sum()
        # add the new job's pt to current time / current available time
        self.available_time = max(self.available_time, self.env.now) + increased_available_time
        self.job_creator.available_time_list[self.m_idx] = self.available_time
        self.que_size = len(self.queue)

    # update the information of progression, eralized and expected tardiness to JOB_CREATOR !!!
    def state_update_global_info_progression(self):
        realized = self.time_till_due.clip(0,1) # realized: 0 if already tardy
        exp = self.slack.clip(0,1) # exp: 0 if slack time is negative
        # update the machine's corresponding record in job creator, and several rates
        self.job_creator.comp_rate_list[self.m_idx] = self.completion_rate
        self.job_creator.comp_rate = np.concatenate(self.job_creator.comp_rate_list).mean()
        self.job_creator.realized_tard_list[self.m_idx] = realized
        self.job_creator.realized_tard_rate = 1 - np.concatenate(self.job_creator.realized_tard_list).mean()
        self.job_creator.exp_tard_list[self.m_idx] = exp
        self.job_creator.exp_tard_rate = 1 - np.concatenate(self.job_creator.exp_tard_list).mean()
        self.job_creator.available_time_list[self.m_idx] = self.available_time

    # update the information of the job that being processed to JOB_CREATOR !!!
    def state_update_global_info_anticipation(self,pt):
        # the index of job that being processed
        current_j_idx = self.queue[self.position]
        self.job_creator.current_j_idx_list[self.m_idx] = current_j_idx
        # next machine of the job
        next_machine = self.sequence_list[self.position][0] if len(self.sequence_list[self.position]) else -1
        self.job_creator.next_machine_list[self.m_idx] = next_machine # update the next wc info (hold by job creator)
        # processing time on next machine
        next_pt = self.remaining_pt_list[self.position][1] if len(self.sequence_list[self.position]) else 0
        self.job_creator.next_pt_list[self.m_idx] = next_pt
        # when the machine is released (sometimes equals next decision point)
        self.release_time = self.env.now + pt
        self.job_creator.release_time_list[self.m_idx] = self.release_time # update the time of completion of current operation
        # the remaining processing time of job upon its completion
        job_rempt = self.remaining_pt_list[self.position].sum() - pt
        self.job_creator.arriving_job_rempt_list[self.m_idx] = job_rempt # update the remaining pt of job under processing
        # the slack time of job
        job_slack = self.slack[self.position]
        self.job_creator.arriving_job_slack_list[self.m_idx] = job_slack # update the slack time of job under processing

    # MUST !!! call this after operation otherwise the record persists and lead to error
    def state_update_global_info_after_operation(self):
        # after each operation, clear the record in job creator, otherwise this value would only changed by next operation
        self.job_creator.next_machine_list[self.m_idx] = -1

    # give ou the information related to sequencing decision
    def sequencing_data_generation(self):
        self.sequencing_data = \
        [self.current_pt, self.remaining_job_pt, np.array(self.due_list), \
        self.env.now, self.time_till_due, self.slack, self.winq, self.avlm, \
        self.next_pt, self.remaining_no_op, self.waited_time, self.queue, self.m_idx]
        return self.sequencing_data


    '''
    4. downwards are functions related to the calculation of reward and construction of state
       only be called if the sequencing learning mode is activated
       the options of reward function are listed at bottom
    '''


    def complete_experience(self):
        # it's possible that not all machines keep memory for learning
        # machine that needs to keep memory don't keep record for all jobs
        # only when they have to choose from several queuing jobs
        try:
            # check whether corresponding experience exists, if not, ends at this line
            self.job_creator.incomplete_rep_memo[self.m_idx][self.decision_point]
            #print('PARAMETERS',self.m_idx,self.decision_point,self.env.now)
            #print('BEFORE\n',self.job_creator.incomplete_rep_memo[self.m_idx][self.decision_point])
            # if yes, get the global state
            sqc_data = self.sequencing_data_generation()
            s_t = self.build_state(sqc_data)
            #print(self.m_idx,s_t)
            r_t = self.reward_function() # can change the reward function, by sepecifying before the training
            self.update_reward_record(r_t)
            #print(self.env.now, r_t)
            self.job_creator.incomplete_rep_memo[self.m_idx][self.decision_point] += [s_t, r_t]
            #print(self.job_creator.incomplete_rep_memo[self.m_idx])
            #print(self.job_creator.incomplete_rep_memo[self.m_idx][self.decision_point])
            complete_exp = self.job_creator.incomplete_rep_memo[self.m_idx].pop(self.decision_point)
            # and add it to rep_memo
            self.job_creator.rep_memo[self.m_idx].append(complete_exp)
            #print(self.job_creator.rep_memo[self.m_idx])
            #print('AFTER\n',self.job_creator.incomplete_rep_memo[self.m_idx][self.decision_point])
            #print(self.m_idx,self.env.now,'state: ',s_t,'reward: ',r_t)
        except:
            pass

    def record_state(self): # reward is given later, now just record the state
        try:
            # check whether corresponding experience exists, if not, ends at this line
            self.job_creator.incomplete_rep_memo[self.m_idx][self.decision_point]
            #print('BEFORE\n',self.job_creator.incomplete_rep_memo[self.m_idx][self.decision_point])
            # if yes, get the global state
            sqc_data = self.sequencing_data_generation()
            s_t = self.build_state(sqc_data)
            #print(self.env.now, r_t)
            self.job_creator.incomplete_rep_memo[self.m_idx][self.decision_point] += [s_t]
            #print('AFTER\n',self.job_creator.incomplete_rep_memo[self.m_idx][self.decision_point])
        except:
            pass
            #print('job %s, record %s of machine %s do not exist!'%(self.job_idx, self.decision_point, self.m_idx))

    # testing reward function
    def get_reward0(self):
        return

    def get_reward1(self):
        '''1. retrive the production record of job'''
        job_record = self.job_creator.production_record[self.job_idx]
        path = job_record[1]
        queued_time = np.array(job_record[2])
        slack = np.array(job_record[3])
        critical_factor = 1 - slack / (np.absolute(slack) + 100)
        exposure = 0.2 # how much of waiting at succeeding machine is exposure to agent
        '''2. calculate the reward for each agents'''
        # if tardiness is non-zero and waiting time exists, machines in path get punishment
        if self.tardiness and queued_time.sum():
            restructured_wait = queued_time*(1-exposure) + np.append(np.delete(queued_time*exposure,0),0)
            restructured_wait *= critical_factor
            reward = - np.square(restructured_wait / 128).clip(0,1)
            #print(reward)
            reward = torch.FloatTensor(reward)
        else:
            reward = torch.ones(len(queued_time),dtype=torch.float)*0
        '''3. and assign the reward to incomplete experience, make them ready to be learned'''
        for i,m_idx in enumerate(path):
            r_t = reward[i]
            decision_point = job_record[0][i][0]
            try:
                self.job_creator.complete_experience(m_idx, decision_point, r_t)
            except:
                pass

    def get_reward2(self):
        '''1. retrive the production record of job'''
        job_record = self.job_creator.production_record[self.job_idx]
        path = job_record[1]
        queued_time = np.array(job_record[2])
        slack = np.array(job_record[3])
        critical_factor = 1 - slack / (np.absolute(slack) + 80)
        exposure = 0.2 # how much of waiting at succeeding machine is exposure to agent
        '''2. calculate the reward for each agents'''
        # if tardiness is non-zero and waiting time exists, machines in path get punishment
        if self.tardiness and queued_time.sum():
            restructured_wait = queued_time*(1-exposure) + np.append(np.delete(queued_time*exposure,0),0)
            restructured_wait *= critical_factor
            reward = - (restructured_wait / 128).clip(0,1)
            #print(reward)
            reward = torch.FloatTensor(reward)
        else:
            reward = torch.ones(len(queued_time),dtype=torch.float)*0
        '''3. and assign the reward to incomplete experience, make them ready to be learned'''
        for i,m_idx in enumerate(path):
            r_t = reward[i]
            decision_point = job_record[0][i][0]
            try:
                self.job_creator.complete_experience(m_idx, decision_point, r_t)
            except:
                pass

    def get_reward3(self):
        '''1. retrive the production record of job'''
        job_record = self.job_creator.production_record[self.job_idx]
        path = job_record[1]
        queued_time = np.array(job_record[2])
        slack = np.array(job_record[3])
        critical_factor = 1 - slack / (np.absolute(slack) + 80)
        exposure = 0.2 # how much of waiting at succeeding machine is exposure to agent
        '''2. calculate the reward for each agents'''
        # if tardiness is non-zero and waiting time exists, machines in path get punishment
        if self.tardiness and queued_time.sum():
            restructured_wait = queued_time*(1-exposure) + np.append(np.delete(queued_time*exposure,0),0)
            restructured_wait *= critical_factor
            reward = - np.square(restructured_wait / 128).clip(0,1)
            #print(reward)
            reward = torch.FloatTensor(reward)
        else:
            reward = torch.ones(len(queued_time),dtype=torch.float)*0
        '''3. and assign the reward to incomplete experience, make them ready to be learned'''
        for i,m_idx in enumerate(path):
            r_t = reward[i]
            decision_point = job_record[0][i][0]
            try:
                self.job_creator.complete_experience(m_idx, decision_point, r_t)
            except:
                pass

    def get_reward4(self):
        '''1. retrive the production record of job'''
        job_record = self.job_creator.production_record[self.job_idx]
        path = job_record[1]
        queued_time = np.array(job_record[2])
        slack = np.array(job_record[3])
        critical_factor = 1 - slack / (np.absolute(slack) + 90)
        exposure = 0.2 # how much of waiting at succeeding machine is exposure to agent
        '''2. calculate the reward for each agents'''
        # if tardiness is non-zero and waiting time exists, machines in path get punishment
        if self.tardiness and queued_time.sum():
            restructured_wait = queued_time*(1-exposure) + np.append(np.delete(queued_time*exposure,0),0)
            restructured_wait *= critical_factor
            reward = - np.square(restructured_wait / 256).clip(0,1)
            #print(reward)
            reward = torch.FloatTensor(reward)
        else:
            reward = torch.ones(len(queued_time),dtype=torch.float)*0
        '''3. and assign the reward to incomplete experience, make them ready to be learned'''
        for i,m_idx in enumerate(path):
            r_t = reward[i]
            decision_point = job_record[0][i][0]
            try:
                self.job_creator.complete_experience(m_idx, decision_point, r_t)
            except:
                pass
