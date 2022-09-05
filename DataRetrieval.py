# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 18:22:21 2021
@author: Amir Modan
Program containing methods for retrieving data from dataset
"""

import scipy.io
import numpy as np
import os
import time
from tqdm import tqdm
from matplotlib import pyplot as plt

def dataset_mat_CSL(fileAddress,ICE=False):
    # Read .mat file into Dictionary object and extract list of data
    result = None
    if ICE:
        return scipy.io.loadmat(fileAddress)['Data']
    else:
        mat = scipy.io.loadmat(fileAddress)['gestures']
    # train = mat[0:mat.shape[0]-1]
    # test = mat[-1]
    # training_data = None
    # testing_data = None
    # training
    for i in range(mat.shape[0]):
        if result is None:
            result = mat[i][0]
        else:
            result = np.concatenate((result,mat[i][0]),axis=1)

    for a in range(7,result.shape[0],8):
        result[a] = 1e-50
    result = result[~np.all(result == 1e-50, axis=1)]
    result = result.T


    # testing
    # for i in range(test.shape[0]):
    #     if testing_data is None:
    #         testing_data = test[i]
    #     else:
    #         testing_data = np.concatenate((testing_data,test[i][0]),axis=1)

    # for a in range(7,testing_data.shape[0],8):
    #     testing_data[a] = 1e-50
    # testing_data = testing_data[~np.all(testing_data == 1e-50, axis=1)]
    return result
    # return result.T
    # return training_data.T,testing_data.T

