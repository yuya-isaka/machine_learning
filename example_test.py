import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

test = pd.read_csv('fresh_aged_ieice/s1.csv', header=None)

print(test)
pred = KMeans(n_clusters=2).fit_predict(test)
print(pred)

print(len(pred))