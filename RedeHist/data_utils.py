# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 08:56:50 2023

@author: 32792
"""



import torch
import numpy as np
import re
import cv2


class data_utils(torch.utils.data.Dataset):
    
    def __init__(self, img_path, mask_path, st_data, align_path, 
                 max_cell = 20, train_resolution=128, index=None):
        super(data_utils, self).__init__()
        
        print("Preparing data ...")
        
        self.max_cell = max_cell
        self.train_resolution = train_resolution
        

        self.align = []
        with open(align_path) as f:
            for line in f:
                if line == '\n':
                    self.align.append([])
                else:
                    self.align.append([int(x) for x in re.split(",", line)])
                    

        img = np.load(img_path, allow_pickle=True)
        self.mask = np.int32(np.load(mask_path, allow_pickle=True))
        self.patch = np.float32(img) / 255
        self.exp = np.float32(st_data.to_df())
        
    
        if not index == None:
            self.patch = self.patch[index, :, :, :]
            self.mask = self.mask[index, :, :]
            self.exp = self.exp[index, :]
            self.align = [self.align[x] for x in index]
        
        
        non_zero = []
        for i in range(len(self.align)):
            result = np.isin(self.mask[i,:,:], self.align[i])
            if np.sum(result) == 0:
                continue
            if not len(self.align[i]) == 0:
                non_zero.append(i)
        non_zero = np.asarray(non_zero)
        
        self.patch = self.patch[non_zero, :, :, :]
        self.mask = self.mask[non_zero, :, :]
        self.exp = self.exp[non_zero, :]
        self.align = [self.align[x] for x in non_zero]
        
        self.max_resolution = self.mask[0].shape[0]
        
        self.exp = torch.tensor(self.exp)
        
        print("Finish initialize ...")


    def __getitem__(self, index):
        
        patch_this = self.patch[index]
        mask_this = self.mask[index]
        exp_this = self.exp[index]
        align = self.align[index]
        
        result = np.isin(mask_this, align)
        value = np.where(np.sum(result, axis=0) > 0.5)[0]
        left = value[0]
        right = value[-1]
        if left == 0:
            left = 1
        value = np.where(np.sum(result, axis=1) > 0.5)[0]
        up = value[0]
        down = value[-1]
        if up == 0:
            up = 1
        gap = max(right - left, down - up)

        res = np.random.randint(gap+1,self.max_resolution)
        choose_x = np.random.randint(max(0,right-res),min(left,self.max_resolution-res))
        choose_y = np.random.randint(max(0,down-res),min(up,self.max_resolution-res))

        #patch_this = patch_this[(choose_x):(choose_x+res), (choose_y):(choose_y+res), :]
        #mask_this = mask_this[(choose_x):(choose_x+res), (choose_y):(choose_y+res)]

        #patch_this = np.float32(cv2.resize(patch_this, (self.train_resolution,self.train_resolution), dst=None, fx=None, fy=None, interpolation=None)).transpose([2,0,1])
        #mask_this = np.float32(cv2.resize(np.float32(mask_this), (self.train_resolution,self.train_resolution), dst=None, fx=None, fy=None, interpolation=None))
        
        patch_this = np.float32(patch_this).transpose([2,0,1])
        mask_this = np.float32(mask_this)

        #if np.sum(mask_this) < 0.5:
        #    mask_this[int(self.train_resolution/2),int(self.train_resolution/2)] = 1


        mask = []
        ave_mask = []
        select = []
        ind = list(range(len(align)))
        np.random.shuffle(ind)

        for i in range(self.max_cell):
            if i == self.max_cell:
                break
            if i < len(align):
                temp = mask_this.copy()
                temp[temp == align[ind[i]]] = 10000000
                temp[temp < 9999999] = 0
                temp[temp > 9999999] = 1
                mask.append(temp)
                ave_mask.append(np.sum(temp))
            else:
                mask.append(np.float32(np.zeros(temp.shape)))
                ave_mask.append(0.0)

        mask = torch.from_numpy(np.stack(mask))
        ave_mask = torch.from_numpy(np.stack(ave_mask))
        if torch.sum(ave_mask) == 0:
            ave_mask = torch.mean(ave_mask).float()
        else:
            ave_mask = torch.mean(ave_mask[ave_mask > 0]).float()
        patch_this = torch.from_numpy(patch_this)

        
         
        return(patch_this, mask, exp_this, index)

    def __len__(self):
        return self.patch.shape[0]
    
    
    
    
    
    
    
    

