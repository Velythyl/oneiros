
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import numpy as np


class dataloader():
    def __init__(self,folder_path,batch_size=50,env_num=3, test_percent = 0.1):
        self.batch_size = batch_size
        self.folder_path = folder_path
        self.files = os.listdir(self.folder_path )

        # WIP, lots of assumption here        
        self.num_sets = len(self.files)/3
        self.current_set=0
        self.env_num = env_num
        self.test_percent = test_percent


    def createloader(self):
        
        # WIP
        if self.current_set == self.num_sets:
            return print("All the files are read")

        data = self.loaddata()
        # data = data.movedim(0,1) #For the dataloader, 50 samples batch
        self.current_set += 1

        # import pdb; pdb.set_trace()

        test_num = np.ceil(data.shape[0]*self.test_percent)
        train_num = data.shape[0] - test_num

        train_set, test_set = random_split(data, [int(train_num), int(test_num)])
        
        train_set = train_set.dataset.movedim(0,1)
        test_set = test_set.dataset.movedim(0,1)

        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=0)   
        return train_loader, test_loader
    
    def loaddata(self):
        datalist = []
        for file in self.files:
            tempdata = torch.load(os.path.join(self.folder_path,file))
            datalist.append(tempdata)
        # for i in range(self.env_num):
        #     idx = i+(self.env_num*self.current_set)
        #     tempdata = torch.load(os.path.join(self.folder_path,self.files[idx]))
        #     datalist.append(tempdata)

        data = torch.cat(datalist,dim=0)
        return data

