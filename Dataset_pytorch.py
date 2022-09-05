from torch.utils.data import Dataset
import numpy as np
import torch
from scipy.ndimage.filters import median_filter
from scipy import signal
from matplotlib import pyplot as plt
class Dataset_pytorch(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data,label,mean=None,std=None):
        # self.data = train_data.reshape(-1,8) if train else test_data.reshape(-1,8)
        # self.data = data.reshape(1,data.shape[0],data.shape[1])
        # self.data = data.reshape(-1,1,24,7)
        # self.data = None
        # for i in range(0,data.shape[0],12):
        #     if self.data is None:
        #         self.data = data[i:i+12].reshape(1,12,168)
        #     else:
        #         self.data = np.concatenate((self.data,data[i:i+12].reshape(1,12,168)),axis=0)
        self.data = data
        
        # self.data = self.NormalizeData(self.data,mean,std)
        # self.data = data        
        self.label = label
        # self.label = train_label if train else test_label

    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        data = self.data[idx]


        data = data.reshape(1,8,24)
        # data = median_filter(data,3)
        data = data.astype("float32")
        return data,self.label[idx]
        # label = [0 for _ in range(4)]
        # label[int(self.label[idx])] = 1
        
    
    def NormalizeData(self,data,mean,std):
        # return (data - (-5.671770187091746e-06)) / (0.003034655367424172)
        return (data - (mean) / (std))
        # return (data - np.min(data)) / ((np.max(data) - np.min(data)))