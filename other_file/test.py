import sys, os
sys.path.append(os.pardir)
import pandas as pd
import numpy as np

test = pd.read_csv('fresh_aged_ieice/s1.csv', header=None)

print(test)
print(type(test))

test_value= test.values
print(test_value)
print(test_value.shape)
a = test_value[np.nonzero(test_value)]
print(len(a))