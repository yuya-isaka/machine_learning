import sys, os
sys.path.append(os.pardir)
import pandas as pd

test = pd.read_csv('fresh_aged_ieice/s1.csv', header=None)

print(test)
print(type(test))

test_value= test.values
print(test_value)
print(test_value.shape)