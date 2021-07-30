import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
# 導入Sklearn套件
# 導入將數據集拆成訓練集與測試集的套件
from sklearn.model_selection import train_test_split
# 導入迴歸模型套件
from sklearn.linear_model import LinearRegression
# 導入多項式套件，建構多項式迴歸模型所需的套件
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from joblib import dump, load
import numpy as np
import argparse

times = 1000
acc = -0.8
parser = argparse.ArgumentParser()
parser.add_argument("TIMES", nargs='?', default=times)
parser.add_argument("ACC", nargs='?', default=acc)
args = parser.parse_args()
times = int(args.TIMES)
acc = float(args.ACC)

x1 = pd.read_csv('rsrp_p0.csv', header=None)
x2 = pd.read_csv('rsrp_p1.csv', header=None)

y1 = pd.read_csv('power_p0.csv', header=None)
y2 = pd.read_csv('power_p1.csv', header=None)

x = pd.concat([x1,x2],ignore_index=True)
y = pd.concat([y1,y2],ignore_index=True)
plt.scatter(x,y,s=25,c='pink')
# for i in range(len(x)):
#     try:
#         temp = x[x[0]==x[0][i]].index
#         if len(temp) == 1:
#             continue
#         value = [y[0][j] for j in temp]
#         bigger = int(np.argmax(value))
#         temp = temp.drop(temp[bigger])
#         x = x.drop(temp)
#         y = y.drop(temp)
#     except:
#         continue
x = x.reset_index(drop = True)
y = y.reset_index(drop = True)
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

color = ['blue','green','black','olive','purple','orange','chocolate']
n = 3
plots = []
f = open('pic_coef\\test_{}_{}.txt'.format(str(acc),str(times)), 'w')
for item in color:
    temp = None
    mini = -2
    poly_reg = PolynomialFeatures(degree=n)
    pol_reg = LinearRegression(fit_intercept = True)
    for i in range(times):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = None)
        X_poly = poly_reg.fit_transform(x_train)
        pol_reg.fit(X_poly, y_train)
        x_test = x.sort_values(by = 0)
        y_pred = pol_reg.predict(poly_reg.fit_transform(x_test))
        score = pol_reg.score(poly_reg.fit_transform(x_test), y)
        if score>acc:
            break
        if score>mini:
            temp = pol_reg
            mini = score
        if i==times-1 and temp != None:
            pol_reg = temp
            score = mini
    print(score)
    dump(pol_reg,'models\\n_{}_{}_{}.joblib'.format(n,acc,times))
    y_pred = pol_reg.predict(poly_reg.fit_transform(x))
    dif = [(y[0][i]-y_pred[i][0]) for i in range(len(x))]
    max = np.max(dif)

    x_all = x.sort_values(by = 0)
    x_all = x_all.reset_index(drop=True)
    y_pred = pol_reg.predict(poly_reg.fit_transform(x_all))
    data, = plt.plot(x_all,y_pred,item,label = 'n = '+str(n))
    plots.append(data,)
    
    f.write("n = {} ".format(str(n)))
    f.write('score = {} '.format(str(score)))
    f.write('max error = {}\ncoef = {}'.format(str(max),'{'))
    for i in pol_reg.coef_[0]:
        f.write(str(i)+', ')
    f.write('{} \nintercept = {}\n'.format('}',pol_reg.intercept_))
    n+=2
f.close()

# print(poly_reg.fit_transform([[18]]))
# print(pol_reg.predict(poly_reg.fit_transform([[18]])))

plt.scatter(x,y,s=25,c='r')
plt.xlabel("RSRP", fontsize=10) 
plt.ylabel("POWER", fontsize=10)
x_major_locator=MultipleLocator(5)
y_major_locator=MultipleLocator(1)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
plt.legend(handles=plots)
plt.savefig('pic_coef\\test_{}_{}.png'.format(str(acc),str(times)), bbox_inches='tight',dpi=150)
plt.show()