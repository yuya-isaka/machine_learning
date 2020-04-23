import numpy as np #適当な配列作るためにNumpy使う
import pandas as pd

test = pd.read_csv('fresh_aged_ieice/s1.csv', header=None)
test_value = np.array(test.values)



X = np.array([i for i in range(0,33)]) #自然数の配列
Y = np.array([i for i in range(0,148)]) #特に意味のない正弦
x = np.arange(0,33,1)
y = np.arange(0,148,1)

z = test_value[y][x]