from time import process_time_ns
import random
import numpy as np

'''
This module contains s series of job sequencing rules
For the purpose of validation and comparison
'''

# Benchmark, as the worst possible case
def random(data):
    job_position = np.random.randint(len(data[0]))
    return job_position, False, 0

def SPT(data): # shortest processing time
    job_position = np.argmin(data[0])
    return job_position, False, 0

def LPT(data): # longest processing time
    job_position = np.argmax(data[0])
    return job_position, False, 0

def LRO(data): # least remaining operations / highest completion rate
    job_position = np.argmax(data[9])
    return job_position, False, 0

def LWKR(data): # least work remaining
    job_position = np.argmin(data[1])
    return job_position, False, 0

def LWKRSPT(data): # remaining work + SPT
    job_position = np.argmin(data[0] + data[1])
    return job_position, False, 0

def LWKRMOD(data): # remaining work + MOD
    due = data[2]
    operational_finish = data[0] + data[3]
    MOD = np.max([due,operational_finish],axis=0)
    job_position = np.argmin(data[0] + data[1] + MOD)
    return job_position, False, 0

def EDD(data):
    # choose the job with earlist due date
    job_position = np.argmin(data[2])
    return job_position, False, 0

def COVERT(data): # cost over time
    average_pt = data[0].mean()
    cost = (data[2] - data[3] - data[0]).clip(0,None)
    priority = (1 - cost / (0.05*average_pt)).clip(0,None) / data[0]
    job_position = priority.argmax()
    return job_position, False, 0

def CR(data):
    CR = data[4] / data[1]
    job_position = CR.argmin()
    return job_position, False, 0

def CRSPT(data): # CR+SPT
    CRSPT = data[4] / data[1] + data[0]
    job_position = CRSPT.argmin()
    return job_position, False, 0

def MS(data):
    slack = data[5]
    job_position = slack.argmin()
    return job_position, False, 0

def MDD(data): # The modified due date is a job's original due date or its early finish time, whichever is larger
    due = data[2]
    finish = data[1] + data[3]
    MDD = np.max([due,finish],axis=0)
    job_position = MDD.argmin()
    return job_position, False, 0

def MON(data):
    # Montagne's heuristic, this rule combines SPT with additional slack factor
    due_over_pt = np.array(data[2])/np.sum(data[0])
    priority = due_over_pt/np.array(data[0])
    job_position = priority.argmax()
    return job_position, False, 0

def MOD(data): # The modified operational due date
    due = data[2]
    operational_finish = data[0] + data[3]
    MOD = np.max([due,operational_finish],axis=0)
    job_position = MOD.argmin()
    return job_position, False, 0

def NPT(data): # next processing time
    job_position = np.argmin(data[9])
    return job_position, False, 0

def ATC(data): # http://www.growingscience.com/ijiec/Vol7/IJIEC_2015_23.pdf
    #print(data)
    average_pt = data[0].mean()
    cost = (data[2] - data[3] - data[0]).clip(0,None)
    #print(average_pt, AT)
    priority = np.exp( - cost / (0.05*average_pt)) / data[0]
    #print(priority)
    job_position = priority.argmax()
    return job_position, False, 0

def AVPRO(data): # average processing time per operation
    AVPRO = data[1] / (data[9] + 1)
    job_position = AVPRO.argmin()
    return job_position, False, 0

def SRMWK(data): # slack per remaining work, identical to CR
    SRMWK = data[5] / data[1]
    job_position = SRMWK.argmin()
    return job_position, False, 0

def SRMWKSPT(data): # slack per remaining work + SPT, identical to CR+SPT
    SRMWKSPT = data[5] / data[1] + data[0]
    job_position = SRMWKSPT.argmin()
    return job_position, False, 0

def WINQ(data): # WINQ
    job_position = data[6].argmin()
    return job_position, False, 0

def PTWINQ(data): # PT + WINQ
    sum = data[0] + data[6]
    job_position = sum.argmin()
    return job_position, False, 0

def PTWINQS(data): # PT + WINQ + Slack
    sum = data[0] + data[5] + data[6]
    job_position = sum.argmin()
    return job_position, False, 0

def DPTWINQNPT(data): # 2PT + WINQ + NPT
    sum = data[0] + data[6] + data[9]
    job_position = sum.argmin()
    return job_position, False, 0

def DPTLWKR(data): # 2PT + LWKR
    sum = data[0] + data[1]
    job_position = sum.argmin()
    return job_position, False, 0

def DPTLWKRS(data): # 2PT + LWKR + slack
    sum = data[0] + data[1] + data[5]
    job_position = sum.argmin()
    return job_position, False, 0

def FIFO(dummy): # first in, first out, data is not needed
    job_position = 0
    return job_position, False, 0

def GP_S1(data): # genetic programming rule 1
    sec1 = data[0] + data[1]
    sec2 = (data[6]*2-1) / data[0]
    sec3 = (data[6] + data[1] + (data[0]+data[1])/(data[6]-data[1])) / data[0]
    sum = sec1-sec2-sec3
    job_position = sum.argmin()
    return job_position, False, 0

def GP_S2(data): # genetic programming rule 2
    NIQ = len(data[0])
    sec1 = NIQ * (data[0]-1)
    sec2 = data[0] + data[1] * np.max([data[0],data[6]],axis=0)
    sec3 = np.max([data[6],NIQ+data[6]],axis=0)
    sec4 = (data[7]+1+np.max([data[1],np.ones_like(data[1])*(NIQ-1)],axis=0)) * np.max([data[6],data[1]],axis=0)
    sum = sec1 * sec2 + sec3 * sec4
    job_position = sum.argmin()
    return job_position, False, 0
