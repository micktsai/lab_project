import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator
from joblib import dump, load

coef = [0.0, -1.4848989387401445e-06, -5.165303251028003e-07, -7.2674583542395754e-06, -6.963654370020427e-05, -0.00036293792787012456, 5.984190556447007e-05, -4.010361459250293e-06, 1.4110462972693913e-07, -2.7560232161094665e-09, 2.8370661701606493e-11, -1.2036192109651854e-13,]
intercept = float(-100.99408995)
poly_reg = PolynomialFeatures(11)
lr = load('models\\n_15_-0.8_1500.joblib')
x1 = pd.read_csv('rsrp_p0.csv', header=None)
x2 = pd.read_csv('rsrp_p1.csv', header=None)

y1 = pd.read_csv('power_p0.csv', header=None)
y2 = pd.read_csv('power_p1.csv', header=None)

x = pd.concat([x1,x2],ignore_index=True)
y = pd.concat([y1,y2],ignore_index=True)
plt.scatter(x,y,s=15,c='r')
plt.xlabel("RSRP", fontsize=10) 
plt.ylabel("POWER", fontsize=10)

x_all = x.sort_values(by = 0)
x_all = x_all.reset_index(drop=True)
ans = []
for j in range(13,50):
  temp2 = 0
  for i in range(len(coef)):
    temp2 += coef[i]*(j**i)
  temp2 = temp2+intercept
  ans.append(temp2)
# result = lr.predict(poly_reg.fit_transform(x_all))+6
plt.plot([i for i in range(13,50)],ans,'g')
plt.plot([i for i in range(13,50)],[ans[i]+6 for i in range(len(ans))],'y')
plt.show()