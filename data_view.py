import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import pandas as pd

x1 = pd.read_csv('rsrp_p0.csv', header=None).values.tolist()
x2 = pd.read_csv('rsrp_p1.csv', header=None).values.tolist()

y1 = pd.read_csv('power_p0.csv', header=None).values.tolist()
y2 = pd.read_csv('power_p1.csv', header=None).values.tolist()

# x1.extend(x2)
# y1.extend(y2)

plt.scatter(x1,y1,s=15,c='r',label = '0')
plt.scatter(x2,y2,s=10,c='b',label = '1')
plt.xlabel("RSRP", fontsize=10) 
plt.ylabel("POWER", fontsize=10)

x_major_locator=MultipleLocator(2)
y_major_locator=MultipleLocator(1)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
plt.legend()
plt.savefig('total')
plt.show()