from sklearn import datasets
import numpy as np
from kmed import *
iris_data = datasets.load_iris()
target = iris_data.target
iris_data = iris_data['data']
t1,t2 = k_medians(iris_data,3,100)
print(t1)
print(t2)
print(target)
acc_rate = np.sum(t2 == target)/len(target)
print(acc_rate)
