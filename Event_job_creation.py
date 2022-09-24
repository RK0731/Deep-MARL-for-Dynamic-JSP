import numpy as np
import random
import sys
sys.path
from tabulate import tabulate
import matplotlib.pyplot as plt

class creation:
    def __init__ (self, env, span, machine_list, pt_range, due_tightness, E_utliz, **kwargs):
        if 'seed' in kwargs:
            np.random.seed(kwargs['seed'])
            print("Random seed of job creation is fixed, seed: {}".format(kwargs['seed']))
        # environemnt and simulation span
        self.env = env
        self.span = span
        # all machines
        self.m_list = machine_list
        self.no_machines = len(self.m_list) # related to number of operations
        # the dictionary that records the details of operation and tardiness
        self.production_record = {}
        self.tardiness_record = {}
        # record operations for gantt chart
        self.op_record = []
        # range of processing time
        self.pt_range = pt_range
        # calulate the average processing time of a single operation
        self.avg_pt = np.average(self.pt_range) - 0.5
        # tightness factor of jobs
        self.tightness = due_tightness
        # expected utlization rate of machines
        self.E_utliz = E_utliz
        # set a variable to track the number of in-system number of jobs
        self.in_system_job_no = 0
        self.in_system_job_no_dict = {}
        self.index_jobs = 0
        # set lists to track the completion rate, realized and expected tardy jobs in system
        self.comp_rate_list = [[] for m in self.m_list]
        self.comp_rate = 0
        self.realized_tard_list = [[] for m in self.m_list]
        self.realized_tard_rate = 0
        self.exp_tard_list = [[] for m in self.m_list]
        self.exp_tard_rate = 0
        # initialize the information associated with jobs that are being processed
        self.available_time_list = np.array([0 for m in self.m_list])
        self.release_time_list = np.array([self.avg_pt for m in self.m_list])
        self.current_j_idx_list = np.arange(self.no_machines)
        self.next_machine_list = np.array([-1 for m in self.m_list])
        self.next_pt_list = np.array([self.avg_pt for m in self.m_list])
        self.arriving_job_ttd_list = np.array([self.avg_pt*self.no_machines for m in self.m_list])
        self.arriving_job_rempt_list = np.array([0 for m in self.m_list])
        self.arriving_job_slack_list = np.array([0 for m in self.m_list])
        # and create an empty, initial array of sequence
        self.sequence_list = []
        self.pt_list = []
        self.remaining_pt_list = []
        self.create_time = []
        self.due_list = []
        self.schedule = []
        # record the rewards that agents received
        self.reward_record = {}
        for m in self.m_list:
            self.reward_record[m.m_idx] = [[],[]]
        # record the arrival and departure information
        self.arrival_dict = {}
        self.departure_dict = {}
        self.mean_dict = {}
        self.std_dict = {}
        self.expected_tardiness_dict = {}
        # decide the feature of new job arrivals
        # beta is the average time interval between job arrivals
        # let beta equals half of the average time of single operation
        self.beta = self.avg_pt / self.E_utliz
        # number of new jobs arrive within simulation
        self.total_no = np.round(self.span/self.beta).astype(int)
        # the interval between job arrivals by exponential distribution
        self.arrival_interval = np.random.exponential(self.beta, self.total_no).round()
        # dynamically change the random seed to avoid extreme case
        if 'random_seed' in kwargs and kwargs['random_seed']:
            interval = self.span/50
            self.env.process(self.dynamic_seed_change(interval))
        # check if jobs got diferent number of operations
        if 'hetero_len' in kwargs and kwargs['hetero_len']:
            pass
        # even the time interval between job arrivals
        if 'even' in kwargs and kwargs['even']:
            print("EVEN mode ON")
            #print(self.arrival_interval)
            self.arrival_interval = np.ones(self.arrival_interval.size)*self.arrival_interval.mean()
            #print(self.arrival_interval)
        # print the arrivals or not
        self.print = 0 # don't print by default
        if 'print' in kwargs:
            self.print = kwargs['print']
        self.initial_job_assignment()
        # start the new job arrival
        self.env.process(self.new_job_arrival())

    def initial_job_assignment(self):
        sqc_seed = np.arange(self.no_machines)
        for m_idx,m in enumerate(self.m_list): # for each machine
            np.random.shuffle(sqc_seed)
            sqc = np.concatenate([np.array([m_idx]),sqc_seed[sqc_seed != m_idx]]) # let the index of machine of first operation equals index of current machine
            # allocate the job index to corrsponding workcenter's queue
            self.sequence_list.append(sqc)
            # produce processing time of job, get corresponding remaining_pt_list
            ptl = np.random.randint(self.pt_range[0], self.pt_range[1], size = [self.no_machines])
            self.pt_list.append(ptl)
            self.record_job_feature(self.index_jobs,ptl)
            # rearrange the order of ptl to get remaining pt list, so we can simply delete the first element after each stage of production
            remaining_ptl = ptl[sqc]
            self.remaining_pt_list.append(remaining_ptl)
            # produce the due date for job
            avg_pt = ptl.mean()
            due = np.round(avg_pt*self.no_machines*np.random.uniform(1, self.tightness))
            # record the creation time and due date of job
            self.create_time.append(0)
            self.due_list.append(due)
            # update the in-system-job number
            self.record_job_arrival()
            # operation record, path, wait time, decision points, slack change
            self.production_record[self.index_jobs] = [[],[],[],[]]
            '''after creation of new job, add it to machine'''
            m.queue.append(self.index_jobs)
            m.sequence_list.append(np.delete(sqc,0)) # the added sequence is the one without first element, coz it's been dispatched
            m.remaining_pt_list.append(remaining_ptl)
            m.due_list.append(due)
            m.slack_upon_arrival.append(due - self.env.now - remaining_ptl.sum())
            m.arrival_time_list.append(self.env.now)
            # after assigned the initial job to machine, activate its sufficient stock event
            m.sufficient_stock.succeed()
            self.index_jobs += 1 # and update the index of job
            if self.print:
                print("**INITIAL ARRIVAL: Job %s, time:%s, sqc:%s, pt(0->m):%s, due:%s"%(self.index_jobs, self.env.now, sqc_seed ,ptl, due))

    def new_job_arrival(self):
        # main process
        sqc_seed = np.arange(self.no_machines)
        while self.index_jobs < self.total_no:
            # draw the time interval betwen job arrivals from exponential distribution
            # The mean of an exp random variable X with rate parameter λ is given by:
            # 1/λ (which equals the term "beta" in np exp function)
            time_interval = self.arrival_interval[self.index_jobs]
            yield self.env.timeout(time_interval)
            # produce sequence of job, first shuffle the sequence seed
            np.random.shuffle(sqc_seed)
            self.sequence_list.append(sqc_seed.copy())
            # produce processing time of job, get corresponding remaining_pt_list
            ptl = np.random.randint(self.pt_range[0], self.pt_range[1], size = [self.no_machines])
            self.pt_list.append(ptl)
            self.record_job_feature(self.index_jobs,ptl)
            # rearrange the order of ptl to get remaining pt list, so we can simply delete the first element after each stage of production
            remaining_ptl = ptl[sqc_seed]
            self.remaining_pt_list.append(remaining_ptl.copy())
            # produce due date for job
            avg_pt = ptl.mean()
            due = np.round(avg_pt*self.no_machines*np.random.uniform(1, self.tightness) + self.env.now)
            # record the creation time
            self.create_time.append(self.env.now)
            self.due_list.append(due)
            # add job to system and create the data repository for job
            self.record_job_arrival()
            # operation record, path, wait time, decision points, slack change
            self.production_record[self.index_jobs] = [[],[],[],[]]
            '''after creation of new job, add it to machine'''
            # first machine of that job
            first_m = sqc_seed[0]
            # add job to machine
            self.m_list[first_m].queue.append(self.index_jobs)
            self.m_list[first_m].sequence_list.append(np.delete(sqc_seed,0)) # the added sequence is the one without first element, coz it's been dispatched
            self.m_list[first_m].remaining_pt_list.append(remaining_ptl)
            self.m_list[first_m].due_list.append(due)
            self.m_list[first_m].slack_upon_arrival.append(due - self.env.now - remaining_ptl.sum())
            self.m_list[first_m].arrival_time_list.append(self.env.now)
            # and update some information
            self.m_list[first_m].state_update_after_job_arrival(remaining_ptl[0])
            # after assigned the nwe job to machine, try activate its sufficient stock event
            try:
                self.m_list[first_m].sufficient_stock.succeed()
            except:
                pass
            self.index_jobs += 1 # and update the index of job
            if self.print:
                print("**ARRIVAL: Job %s, time:%s, sqc:%s, pt(0->m):%s, due:%s"%(self.index_jobs, self.env.now, sqc_seed ,ptl, due))

    def dynamic_seed_change(self, interval):
        while self.env.now < self.span:
            yield self.env.timeout(interval)
            seed = np.random.randint(2000000000)
            np.random.seed(seed)
            print('change random seed to {} at time {}'.format(seed,self.env.now))

    def change_setting(self,pt_range):
        print('Heterogenity changed at time',self.env.now)
        self.pt_range = pt_range
        self.avg_pt = np.average(self.pt_range)-0.5
        self.beta = self.avg_pt / (2*self.E_utliz)

    def get_global_exp_tard_rate(self):
        x = []
        for m in self.m_list:
            x = np.append(x, m.slack)
        rate = x[x<0].size / x.size
        return rate

    # this fucntion record the time and number of new job arrivals
    def record_job_arrival(self):
        self.in_system_job_no += 1
        self.in_system_job_no_dict[self.env.now] = self.in_system_job_no
        try:
            self.arrival_dict[self.env.now] += 1
        except:
            self.arrival_dict[self.env.now] = 1

    # this function is called upon the completion of a job, by machine agent
    def record_job_departure(self):
        self.in_system_job_no -= 1
        self.in_system_job_no_dict[self.env.now] = self.in_system_job_no
        try:
            self.departure_dict[self.env.now] += 1
        except:
            self.departure_dict[self.env.now] = 1

    def record_job_feature(self,idx,ptl):
        self.mean_dict[idx] = (self.env.now, ptl.mean())
        self.std_dict[idx] = (self.env.now, ptl.std())

    def build_sqc_experience_repository(self,m_list): # build two dictionaries
        self.incomplete_rep_memo = {}
        self.rep_memo = {}
        for m in m_list: # each machine will have a list in the dictionaries
            self.incomplete_rep_memo[m.m_idx] = {} # used for storing s0 and a0
            self.rep_memo[m.m_idx] = [] # after the transition and reward is given, store r0 and s1, complete the experience

    def complete_experience(self, m_idx, decision_point, r_t): # turn incomplete experience to complete experience
        self.incomplete_rep_memo[m_idx][decision_point] += [r_t]
        complete_exp = self.incomplete_rep_memo[m_idx].pop(decision_point)
        self.rep_memo[m_idx].append(complete_exp)
        self.reward_record[m_idx][0].append(self.env.now)
        self.reward_record[m_idx][1].append(r_t)

    def initial_output(self):
        print('job information are as follows:')
        job_info = [[i,self.sequence_list[i], self.pt_list[i], \
        self.create_time[i], self.due_list[i]] for i in range(self.index_jobs)]
        print(tabulate(job_info, headers=['idx.','sqc.','proc.t.','in','due']))
        print('--------------------------------------')
        return job_info

    def final_output(self):
        # information of job output time and realized tardiness
        output_info = []
        print(self.production_record)
        for item in self.production_record:
            output_info.append(self.production_record[item][4])
        job_info = [[i,self.sequence_list[i], self.pt_list[i], self.create_time[i],\
        self.due_list[i], output_info[i][0], output_info[i][1]] for i in range(self.index_jobs)]
        print(tabulate(job_info, headers=['idx.','sqc.','proc.t.','in','due','out','tard.']))
        realized = np.array(output_info)[:,1].sum()
        exp_tard = sum(self.expected_tardiness_dict.values())

    def tardiness_output(self):
        # information of job output time and realized tardiness
        tard_info = []
        #print(self.production_record)
        for item in self.production_record:
            #print(item,self.production_record[item])
            tard_info.append(self.production_record[item][4])
        # now tard_info is an ndarray of objects, cannot be sliced. need covert to common np array
        # if it's a simple ndarray, can't sort by index
        dt = np.dtype([('output', float),('tardiness', float)])
        tard_info = np.array(tard_info, dtype = dt)
        tard_info = np.sort(tard_info, order = 'output')
        # now tard_info is an ndarray of objects, cannot be sliced, need covert to common np array
        tard_info = np.array(tard_info.tolist())
        tard_info = np.array(tard_info)
        output_time = tard_info[:,0]
        tard = np.absolute(tard_info[:,1])
        cumulative_tard = np.cumsum(tard)
        tard_max = np.max(tard)
        tard_mean = np.cumsum(tard) / np.arange(1,len(cumulative_tard)+1)
        tard_rate = tard.clip(0,1).sum() / tard.size
        #print(output_time, cumulative_tard, tard_mean)
        return output_time, cumulative_tard, tard_mean, tard_max, tard_rate

    def record_printout(self):
        print(self.production_record)

    def timing_output(self):
        return self.arrival_dict, self.departure_dict, self.in_system_job_no_dict

    def feature_output(self):
        return self.mean_dict, self.std_dict

    def reward_output(self, m_idx):
        plt.scatter(np.array(self.reward_record[m_idx][0]), np.array(self.reward_record[m_idx][1]), s=3, c='r')
        plt.show()
        return

    def all_tardiness(self):
        # information of job output time and realized tardiness
        tard = []
        #print(self.production_record)
        for item in self.production_record:
            #print(item,self.production_record[item])
            tard.append(self.production_record[item][4][1])
        #print(tard)
        tard = np.array(tard)
        mean_tardiness = tard.mean()
        tardy_rate = tard.clip(0,1).sum() / tard.size
        #print(output_time, cumulative_tard, tard_mean)
        return mean_tardiness, tardy_rate
