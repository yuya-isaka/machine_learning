import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd

test = pd.read_csv('fresh_aged_ieice/s1.csv', header=None)
test2 = pd.read_csv('fresh_aged_ieice/s1_aged.csv', header=None)
frequency = test.values
frequency2 = test2.values

x = np.empty([148, 33])
y = np.empty([148, 33])

for i in range(148):
    for j in range(33):
        x[i,j] = j
        y[i,j] = i

plt.subplot(2,1,1)
plt.contourf(x,y,frequency)
plt.colorbar()

plt.subplot(2,1,2)
plt.contourf(x,y,frequency2)
plt.colorbar()
plt.show()

