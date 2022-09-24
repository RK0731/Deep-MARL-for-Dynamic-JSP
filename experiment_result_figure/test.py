import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.ticker as mtick

x= [[1,2,3,4,5,6,7],[2,3,4,5,6,7,8]]

plt.boxplot(x)
plt.show()
