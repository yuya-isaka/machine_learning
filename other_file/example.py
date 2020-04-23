import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

test = pd.read_csv("http://pythondatascience.plavox.info/wp-content/uploads/2016/05/Wholesale_customers_data.csv")

print(test) #(440,種類数)

del(test['Channel'])
del(test['Region'])

test_array = np.array([test['Fresh'].tolist(),
                       test['Milk'].tolist(),
                       test['Grocery'].tolist(),
                       test['Frozen'].tolist(),
                       test['Detergents_Paper'].tolist(),
                       test['Delicassen'].tolist()],
                       np.int32)

test_array = test_array.T
print(test_array) #(440,種類数-2)

pred = KMeans(n_clusters=4).fit_predict(test_array)
print(pred)