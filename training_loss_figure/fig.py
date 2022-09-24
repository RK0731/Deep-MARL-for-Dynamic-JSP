import matplotlib.pyplot as plt
import sys
import numpy as np


fig = plt.figure(figsize=(11,4.5))
titles = ['(a) Utilization rate = 70 %', '(b) Utilization rate = 80 %', '(c) Utilization rate = 90 %']

for ax_idx in range(3):
    with open('training_loss_figure//training_loss_record_{}.txt'.format(ax_idx+1),'r') as f:
        lines = f.readlines()
    data = eval(lines[0])

    # left half, showing the loss of training
    axe = fig.add_subplot(1,3,ax_idx+1)
    iterations = np.arange(len(data))
    axe.scatter(iterations, data,s=0.6,color='b', alpha=0.2)
    # moving average
    x = 50
    axe.plot(np.arange(x/2,len(data)-x/2+1,1),np.convolve(data, np.ones(x)/x, mode='valid'),color='navy',label='moving average',zorder=3)
    # limits, grids,
    ylim_upper = 0.2
    ylim_lower = 0.05
    axe.set_xlim(0,len(data))
    axe.set_ylim(ylim_lower,ylim_upper)
    xtick_interval = 1000
    axe.set_xticks(np.arange(0,len(data)+1,xtick_interval))
    axe.set_xticklabels(np.arange(0,len(data)/xtick_interval,1).astype(int),rotation=45, ha='right', rotation_mode="anchor", fontsize=8)
    axe.set_yticks(np.arange(ylim_lower, ylim_upper+0.01, 0.01))
    axe.grid(axis='x', which='major', alpha=0.5, zorder=0, )
    axe.grid(axis='y', which='major', alpha=0.5, zorder=0, )
    axe.legend()
    # dual axis
    ax_time = axe.twiny()
    ax_time.set_xlim(10000,100000)
    ax_time.set_xticks(np.arange(10000,100000+1,xtick_interval*10))
    ax_time.set_xticklabels(np.arange(10000/xtick_interval,100000/xtick_interval+1,10).astype(int),rotation=45, ha='left', rotation_mode="anchor", fontsize=8)
    ax_time.set_xlabel(titles[ax_idx])

    if ax_idx == 0:
        axe.set_ylabel('Loss (error) of training', fontsize = 10)
    if ax_idx == 1:
        axe.set_xlabel('Iterations of training ('+r'$\times 10^3$'+')', fontsize = 10)
    if ax_idx != 0:
        axe.axes.yaxis.set_ticklabels([])
fig.suptitle('Time in simulation ('+r'$\times 10^3$'+', excluding warm up phase)', fontsize = 10)

fig.subplots_adjust(top=0.8, bottom=0.1, wspace=0.05)
plt.show()


# save the figure if required
address = sys.path[0]+"\\training_loss.png"
fig.savefig(address, dpi=600, bbox_inches='tight')
print('figure saved to'+address)
