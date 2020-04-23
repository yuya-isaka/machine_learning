import pickle
from statistics import mean
import pandas as pd
import numpy as np

f = open('acn_list.binaryfile', 'rb')
acn_list = pickle.load(f)
f.close()

f = open('aged_acn_list.binaryfile', 'rb') 
aged_acn_list = pickle.load(f)
f.close()

index1 = np.arange(1,51)
index2 = np.arange(1,3)
columns1 = ["ACN"]
columns2 = ["aged_ACN"]

test = pd.DataFrame(data=acn_list, index=index1, columns=columns1)
test["FPGA"] = np.arange(1,51)
test.set_index("FPGA",inplace=True)

test2 = pd.DataFrame(data=aged_acn_list, index=index2, columns=columns2)
test2["FPGA"] = np.arange(1,3)
test2.set_index("FPGA",inplace=True)

print(test)
print(test2)
