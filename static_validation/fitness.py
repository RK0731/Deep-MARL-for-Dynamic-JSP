import simpy
import pandas as pd
from pandas import DataFrame


class shopfloor:
    def __init__(self, env, m_no, **kwargs):
        # STEP 1: create environment for simulation
        self.env = env
        self.m_no = m_no
        self.m_list = []

        # STEP 2: create machines as resources
        for i in range(1,m_no+1):
            expr1 = '''self.m_{} = simpy.Resource(self.env, capacity=1)'''.format(i) # create machines
            exec(expr1)
            expr2 = '''self.m_list.append(self.m_{})'''.format(i) # add to machine list
            exec(expr2)


    def execution(self, schedule):
        while schedule:
            req = self.m_list[schedule.pop(0)].request()
            yield req
            yield self.env.timeout()
            sequence[i].release(req)

    def simulation(self):
        self.env.run()

export_result = 0

schedule = [[1,2,3],[1,2,3],[3,2,1]]
operation_sequence = [[2,3,1],[1,2,3],[3,2,1]]
processing_time = [[4,6,1],[5,3,1],[6,4,2]]
# an extra run with DRL
env = simpy.Environment()
spf = shopfloor(env, 3)
spf.reset(schedule, operation_sequence, processing_time)
spf.simulation()
