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

# data = []
# for i in range(50):
#     data.append(np.ndarray.flatten(residual_data[i]))
# data = np.array(data)

aged_data = []
for i in range(2):
    aged_data.append(np.ndarray.flatten(aged_residual_data[i]))
aged_data = np.array(aged_data)

print(aged_data.shape)

# fig,ax = plt.subplots(5,10,figsize=(20,10))
# count = 0
# for i in range(5):
#     for j in range(10):
#         dummy = sns.distplot(data[count], ax=ax[i, j])
#         count += 1
# plt.show()

fig,ax = plt.subplots(1, 2, figsize=(20, 10))
count = 0
for i in range(1):
    for j in range(2):
        dummy = sns.distplot(aged_data[count], ax=ax[j])
        count += 1
plt.show()