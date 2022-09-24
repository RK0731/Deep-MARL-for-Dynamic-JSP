import numpy as np
import random
import sys
import pandas as pd
from pandas import DataFrame
from tabulate import tabulate

class creation:
    def __init__(self, m_no, j_no, pt_range, tightness):
        self.m_no = m_no
        self.j_no = j_no
        self.operation_sequence = []
        self.processing_time = []
        self.due_date = []
        for i in range(self.j_no):
            # operations
            sqc_seed = np.arange(self.m_no)
            np.random.shuffle(sqc_seed)
            self.operation_sequence.append(sqc_seed.tolist())
            # processing time of job
            ptl = np.random.randint(pt_range[0], pt_range[1], size = [m_no])
            self.processing_time.append(ptl.tolist())
            # produce due date for job
            due = np.round(ptl.sum()*np.random.uniform(1, tightness))
            self.due_date.append(due)

    def output(self):
        return self.operation_sequence, self.processing_time, self.due_date

x = creation(10,10,[1,50],3)
operation_sequence, processing_time, due_date = x.output()

print(operation_sequence)
print(processing_time)
print(due_date)

instance = open("test.txt","w")
for L in [operation_sequence, processing_time, due_date]:
    instance.write(str(L))
    instance.write('\n')
instance.close()
