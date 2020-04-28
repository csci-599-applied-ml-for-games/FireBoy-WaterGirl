import matplotlib.pyplot as plt
import numpy as np
import random
# Some example data to display
x = np.linspace(0, 2 * np.pi, 400)
y = np.sin(x ** 2)

hist=[]
for i in range(100):
    hist.append(random.random())

def moving_average_diff(a, n=1):
    diff = np.diff(a)
    ret = np.cumsum(diff, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

print(1.410395981737071758e-01)

f_1=open('C:/Users/naman/Desktop/moving_avg.txt','r')
values1=[]
for x in f_1:
    values1.append(float(x))
    
f_1=open('C:/Users/naman/Desktop/loss.txt','r')
values2=[]
for x in f_1:
    values2.append(float(x))


f1=plt.figure()
f2=plt.figure()

a=f1.add_subplot(111)
a.plot(range(1,101),values1)
a.set_ylabel('points per # of moves taken')
a.set_xlabel('no of games')

b=f2.add_subplot(111)
b.set_ylabel('loss per game')
b.set_xlabel('no of games')
b.plot(range(1,101),values2)
