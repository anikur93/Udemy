import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

x = []
y = []
for line in open('data_1d.csv'):
    a,b = line.split(',')
    x.append(float(a))
    y.append(float(b))
    
x = np.array(x)
y = np.array(y)

plt.scatter(x,y)
plt.show()

den = (x.mean() * x.sum()) - x.dot(x)

slope = ((y.mean() * x.sum()) - (x.dot(y)))/ den

intp = (x.dot(y) * x.mean())/  den

yhat = slope*x + intp

plt.scatter(x,y)
plt.plot(x, yhat)
plt.show()

d1 = y - yhat
d2 = y - y.mean()
r2 = 1 - (d1.dot(d1)/d2.dot(d2))
print(r2)



