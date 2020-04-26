import sys
import os
sys.path.append(os.pardir)
from dataset.fpga_fp_data import generate_data
import matplotlib.pyplot as plt              
import numpy as np
import pandas as pd
from statistics import mean
import pickle
import generate_nnr_data
import seaborn as sns

residual_data, aged_residual_data = generate_nnr_data.generate_nnr()

data = []
for i in range(50):
    data.append(residual_data.flatten())
for i in range(2):
    data.append(aged_residual_data.flatten())
data = np.array(data)

fig,ax = plt.subplots(6,10,figsize=(20,10))
count = 0
for i in range(6):
    for j in range(10):
        if count >= 52:
            break
        dummy = sns.distplot(data[count], ax=ax[i, j])
        count += 1
    if count >= 52:
        break

plt.show()

# fig,ax = plt.subplots(1, 2, figsize=(20, 10))
# count = 0
# for i in range(1):
#     for j in range(2):
#         dummy = sns.distplot(aged_data[count], ax=ax[j])
#         count += 1
# plt.show()

# weights = np.ones_like(np.array(aged_data[0]))/float(len(np.array(aged_data[0])))

# plt.hist(aged_data[0], weights=weights)
# plt.show()

