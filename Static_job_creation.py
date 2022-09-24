import numpy as np
import random
import sys
from tabulate import tabulate
import matplotlib.pyplot as plt

class creation:
    def __init__ (self, env, machine_list, operation_sequence, processing_time, due_date, **kwargs):
        self.env = env
        self.m_list = machine_list
        self.no_machines= len(self.m_list)
        self.op_sqc = operation_sequence
        self.pt = processing_time
        self.due_date = due_date
        self.pt_range = 'STATIC'
        self.tightness = 'STATIC'
        self.E_utliz = 'STATIC'
        # the dictionary that records the details of operation and tardiness
        self.production_record = {}
        self.tardiness_record = {}
        # record operations for gantt chart
        self.op_record = []
        # set lists to track the completion rate, realized and expected tardy jobs in system
        self.comp_rate_list = [[] for m in self.m_list]
        self.comp_rate = 0
        self.realized_tard_list = [[] for m in self.m_list]
        self.realized_tard_rate = 0
        self.exp_tard_list = [[] for m in self.m_list]
        self.exp_tard_rate = 0
        # initialize the information associated with jobs that are being processed
        self.available_time_list = np.array([0 for m in self.m_list])
        self.release_time_list = np.array([0 for m in self.m_list])
        self.current_j_idx_list = np.arange(self.no_machines)
        self.next_machine_list = np.array([-1 for m in self.m_list])
        self.next_pt_list = np.array([0 for m in self.m_list])
        self.arriving_job_ttd_list = np.array([0*self.no_machines for m in self.m_list])
        self.arriving_job_rempt_list = np.array([0 for m in self.m_list])
        self.arriving_job_slack_list = np.array([0 for m in self.m_list])
        # and create an empty, initial array of sequence
        self.sequence_list = []
        self.pt_list = []
        self.remaining_pt_list = []
        self.due_list = []
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
        # set a variable to track the number of in-system number of jobs
        self.in_system_job_no = 0
        self.in_system_job_no_dict = {}
        self.schedule = []
        self.initial_job_assignment()


    def initial_job_assignment(self):
        for j_idx in range(len(self.due_date)):
            sqc = np.array(self.op_sqc[j_idx])
            self.sequence_list.append(sqc)
            # get processing time of job
            ptl = np.array(self.pt[j_idx])
            self.pt_list.append(ptl)
            # rearrange the order of ptl to get remaining pt list, so we can simply delete the first element after each stage of production
            remaining_ptl = ptl
            self.remaining_pt_list.append(remaining_ptl)
            # produce the due date for job
            due = np.array(self.due_date[j_idx])
            # record the creation time and due date of job
            self.due_list.append(self.due_date[j_idx])
            # operation record, path, wait time, decision points, slack change
            self.production_record[j_idx] = [[],[],[],[]]
            '''add job to machine'''
            m = self.m_list[sqc[0]]
            m.queue.append(j_idx)
            m.sequence_list.append(np.delete(sqc,0)) # the added sequence is the one without first element, coz it's been dispatched
            m.remaining_pt_list.append(remaining_ptl)
            m.due_list.append(due)
            m.slack_upon_arrival.append(due - self.env.now - remaining_ptl.sum())
            m.arrival_time_list.append(self.env.now)
            # after assigned the job to machine, activate its sufficient stock event
            try:
                m.sufficient_stock.succeed()
            except:
                pass


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
