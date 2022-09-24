import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

class draw:
    def __init__(self, data, m_no, due_date):
        self.data = data
        # create the figure instance
        fig = plt.figure(figsize=(10,m_no))
        gantt_chart = fig.add_subplot(1,1,1)
        # vertical position of bars
        bar_pos = np.stack([np.arange(1, m_no + 1)-0.25, np.ones(m_no)/2], axis=1)
        # vertical position of tick labels
        GC_yticks_pos = np.arange(1, m_no + 1)
        # color of bars / jobs
        col_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

        # each item contains information associated with an operation
        # item is a tuple: (job_index, start time, processing time, machine_index)
        for item in self.data:
            # plot blocks (operations)
            gantt_chart.broken_barh([(item[1], item[2])], bar_pos[item[3]], color=col_list[item[0]%10], edgecolor='w')
            # plot the index of jobs
            gantt_chart.text(item[1]+item[2]/2, item[3]+0.95, item[0], fontsize=9, ha='center', va='center', color='w')
        # plot the due dates of job
        height = 0.3
        for idx, time in enumerate(due_date):
            # plot arrows
            gantt_chart.arrow(time, height, 0, -height, width=0.2, head_width=height, head_length=height/2, color=col_list[idx%10], length_includes_head=True)
            # plot the index of jobs
            gantt_chart.text(time, height, idx, fontsize=9, ha='right', color='k')
        # find the output time of last job, determine the range of plot
        last_output = self.data[-1][1] + self.data[-1][2]
        plot_range = np.ceil(last_output/10)*10
        #print(last_output,plot_range)
        # labels, ticks and grids
        gantt_chart.set_xlabel('Time of simulation')
        gantt_chart.set_ylabel('Machines')
        gantt_chart.set_title('The production record (Gantt Chart)')
        gantt_chart.set_yticks(GC_yticks_pos)
        gantt_chart.set_yticklabels([i for i in range(m_no)])
        # set grid and set grid behind bars
        fig_major_ticks = np.arange(0, plot_range+1, 10)
        fig_minor_ticks = np.arange(0, plot_range+1, 1)
        # Major ticks every 20, minor ticks every 5
        gantt_chart.set_xticks(fig_major_ticks)
        gantt_chart.set_xticks(fig_minor_ticks, minor=True)
        # different settings for the grids:
        gantt_chart.grid(which='major', alpha=1)
        gantt_chart.grid(which='minor', alpha=0.2, linestyle='--')
        gantt_chart.set_axisbelow(True)
        # limit
        gantt_chart.set_xlim(0, plot_range)
        gantt_chart.set_ylim(0)
        #gantt_chart.legend()
        fig.subplots_adjust(top=0.9, bottom=0.15, hspace=0.5)
        plt.show()
