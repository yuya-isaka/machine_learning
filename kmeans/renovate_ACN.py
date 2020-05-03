import pickle
import numpy as np
import pandas as pd

f = open('acn_list.binaryfile', 'rb') 
aged_acn_list = pickle.load(f)
f.close()

aged_acn_list = np.array(aged_acn_list)
aged_acn_list= aged_acn_list.T

index_list = np.arange(1, 53)
acn_list = []
average_list = []
status_list = []
for x in aged_acn_list:
    acn = 0
    tmp = 0
    for i in range(len(x)):
        if tmp < x[i]:
            tmp = x[i]
            acn = i + 2

    average_list.append(tmp)
    acn_list.append(acn)

for i in range(50):
    status_list.append('unused')
for i in range(2):
    status_list.append('aged')

print(acn_list)
print(len(acn_list))

test = [average_list, acn_list, status_list]
test = np.array(test)
test = test.T
df = pd.DataFrame(test,
                  columns=['Maximum Average Silhouette value', 'Appropriate Cluster Number', 'Status'],
                  index=index_list)

print(df)
        
