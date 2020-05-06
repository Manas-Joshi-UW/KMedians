import numpy as np
from numpy import random
from geometric_median import geometric_median
import time
import scipy
from scipy.spatial.distance import cdist

# return a matrix where each row is a randomized numpy array
def initialize_k_meds(data, k, dim, high= None, low= None):
    if k is None and dim is None:
        return None
    if k <=0:
        return None
    if dim <= 0:
        return None
    if high is None and low is not None:
        return random.randint(size=(k,dim), low=low)
    if low is None and high is not None:
        return random.randint(size=(k,dim), high=high)
    return data[random.randint(size=k, low=0, high=data.shape[0])]# if both are not none this will run

def dist_metr(a,b):
    return np.sum(np.abs(a-b))


def update_medians(data, lab, k):
    k_meds = np.zeros((k, data.shape[1]))
    for i in range(0, k):
        if data[lab==i,:].shape[0] > 0:
            k_meds[i,:]= geometric_median(data[lab==i,:])
    return k_meds


def k_medians(data, k, max_iter, eps=1e-5):
    if data is None:
        return None

    dim = data.shape[1]
    k_meds = initialize_k_meds(data, k, dim) # intialize the medians
    k_meds_updated = initialize_k_meds(data, k,dim)
    lab = np.zeros((1,data.shape[0])) # intialize the label for each row
    dists = np.zeros((data.shape[0], k)) # will use this to store the distances
    for j in range(0, max_iter):
        for i in range(0, k):
            med = k_meds[i,:]
            dist = np.apply_along_axis(dist_metr, axis=1, arr= data,b= med)
            dists[:,i] = dist #set the column equal to the distance vector
        lab = np.argmin(dists, axis= 1)
        k_meds= update_medians(data, lab, k)
    return [k_meds, lab]
