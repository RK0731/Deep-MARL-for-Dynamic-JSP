import matplotlib.pyplot as plt
fig = plt.figure()

ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(222)
ax2.get_yaxis().set_visible(False)
plt.show()
