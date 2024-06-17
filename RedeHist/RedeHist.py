# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------
#
# File: RedeFISH.py
#
# System:         Linux, Windows
# Component Name: RedeFISH
# Version:        20230906
# Language: python3
# Latest Revision Time: 2023/09/06
#
# License: To-be-decided
# Licensed Material - Property of CPNL.
#
# (c) Copyright CPNL. 2023
#
# Address:
# 28#, ZGC Science and Technology Park, Changping District, Beijing, China
#
# Author: Zhong Yunshan
# E-Mail: 327922729@qq.com
#
# Description: Main function of RedeFISH
#
# Change History:
# Date         Author            Description
# 2023/09/06   Zhong Yunshan     Release v1.0.0
# ------------------------------------------------------------------------------------


import os
import time
import pandas as pd
import numpy as np
from scipy import spatial
import scanpy as sc
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim import lr_scheduler
import torch.nn.functional as F
import random
import pytorch_lightning as pl




class DoubleConv2d(nn.Module):
    def __init__(self, inputChannel, outputChannel):
        super(DoubleConv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inputChannel, outputChannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(outputChannel),
            nn.ReLU(True),
            nn.Conv2d(outputChannel, outputChannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(outputChannel),
            nn.ReLU(True)
        )

    def forward(self, x):
        out = self.conv(x)
        return out


# Down Sampling
class DownSampling(nn.Module):
    def __init__(self):
        super(DownSampling, self).__init__()
        self.down = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        out = self.down(x)
        return out


# Up Sampling
class UpSampling(nn.Module):

    # Use the deconvolution
    def __init__(self, inputChannel, outputChannel):
        super(UpSampling, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(inputChannel, outputChannel, kernel_size=2, stride=2),
            nn.BatchNorm2d(outputChannel)
        )

    def forward(self, x, y):
        x =self.up(x)
        diffY = y.size()[2] - x.size()[2]
        diffX = y.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        out = torch.cat([y, x], dim=1)
        return out
    
    


class RedeHist_model(pl.LightningModule):
    
    def __init__(self, sc_data, max_cell = 20, mode = "deconvolution", device = 'cpu'):
        super().__init__()    
        
        self.max_cell = max_cell

        self.nCount_var = sc_data.n_vars
        self.nCount_obs = sc_data.n_obs
        
        self.mode = mode
        
        
        annotation_class = np.unique(sc_data.obs['annotation'])
        class2index = {}
        for i in range(len(annotation_class)):
            class2index[annotation_class[i]] = i
            
        anno_class = np.zeros((len(sc_data), len(annotation_class)))
        for i in range(len(sc_data.obs['annotation'])):
            anno_class[i, class2index[sc_data.obs['annotation'][i]]] = 1
        anno_class = torch.tensor(anno_class)
        self.anno_class = nn.Parameter(anno_class.float(), requires_grad=False)
        
        
        self.sc_data = torch.from_numpy(np.asarray(sc_data.to_df())).float()
        self.sc_data = self.sc_data / torch.sum(self.sc_data, dim=1)[:,None] * self.sc_data.shape[1]
        self.sc_data = nn.Parameter(self.sc_data, requires_grad=False)
       
        #self.sc_data2 = (self.sc_data - torch.mean(self.sc_data, dim=1)[:,None]) / (torch.std(self.sc_data, dim=1)[:,None] + 1e-7) 
        
        self.sc_drop = nn.Parameter((torch.zeros([self.nCount_obs, self.nCount_var])+1))
        
        self.sc2feature1 = torch.nn.Linear(self.nCount_var, 200)
        self.sc2feature2 = torch.nn.Linear(200, 200)
        self.sc2feature3 = torch.nn.Linear(200, 200)
        
        self.layer1 = DoubleConv2d(3, 64)
        self.layer2 = DoubleConv2d(64, 128)
        self.layer3 = DoubleConv2d(128, 256)
        self.layer4 = DoubleConv2d(256, 512)
        self.layer5 = DoubleConv2d(512, 1024)
        self.layer6 = DoubleConv2d(1024, 512)
        self.layer7 = DoubleConv2d(512, 256)
        self.layer8 = DoubleConv2d(256, 128)
        self.layer9 = DoubleConv2d(128, 64)

        self.layer10 = nn.Conv2d(64, 200, kernel_size=3, padding=1)  # The last output layer

        self.weight = nn.Parameter(torch.ones(20, 1, 3, 3), requires_grad=False)
        self.weight2 = nn.Parameter(torch.ones(1, 1, 3, 3), requires_grad=False)
       
        self.loss_BCE = nn.BCELoss()
        #self.loss_BCE = nn.CrossEntropyLoss()

        self.down = DownSampling()
        self.up1 = UpSampling(1024, 512)
        self.up2 = UpSampling(512, 256)
        self.up3 = UpSampling(256, 128)
        self.up4 = UpSampling(128, 64)
       
        self.out2out = torch.nn.Linear(200, 200) 
        self.feature2mask = torch.nn.Linear(200, 1)
        self.feature2expression = torch.nn.Linear(200, 256)
        
        
        self.mask_mRNA_ratio = nn.Parameter(torch.ones([1]))
        
        self.dropout = nn.Dropout(p=0.3)


        self.layer_norm_pixel = nn.LayerNorm(200)
        self.layer_norm_expression = nn.LayerNorm(200)
        self.layer_norm_256 = nn.LayerNorm(256)
        self.layer_norm_scExp = nn.LayerNorm(self.nCount_var) 
        
        self.gene_feature = nn.Parameter(0.25 * torch.randn([self.nCount_var, 256]))
    
        self.mask2mRNA = nn.Parameter(torch.ones([1]))
        
        self.epoch = 0   
    
        self.some_parameters = [self.gene_feature, self.mask2mRNA, self.sc_drop]
        self.other_parameters = [p for p in self.parameters() if ((p not in set(self.some_parameters)))]
        self.optimizer = torch.optim.Adam([{'params': self.some_parameters, 'lr': 0.001, 'initial_lr':0.001},
                                           {'params': self.other_parameters, 'lr': 0.00005, 'initial_lr': 0.00005}], 
                                          lr=0.00005)
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, 0.99)
        
        
        self.total_loss = []
        self.total_loss_st_sim = []
        self.total_loss_st_sim2 = []
        self.total_loss_trans_sim = []
        self.total_loss_entropy1 = []
        self.total_loss_entropy2 = []
        
        self.real_predict = []
        self.predict_predict = []
        self.real_real = [] 

    def latent_to_expression(self, expression, mRNA, is_train):
        
        if is_train:
            expression = self.dropout(torch.tanh(self.feature2expression(expression)))
            expression = torch.matmul(expression, self.layer_norm_256(self.gene_feature).T) / torch.sqrt(torch.tensor(256))
        else:
            expression = torch.tanh(self.feature2expression(expression))
            expression = torch.matmul(expression, self.layer_norm_256(self.gene_feature).T) / torch.sqrt(torch.tensor(256))
        
        #expression = w * (torch.exp(torch.relu(self.feature2expression4(expression))) - 1)
        expression = torch.exp(expression)
        expression = expression / (torch.sum(expression, dim=-1) + 1e-6).unsqueeze(-1) * mRNA
        
        return(expression)
        
        
    def forward(self, x):
        # use forward for inference/predictions
        
        #x = x.permute(0,3,1,2)
        
        conv1 = self.layer1(x)
        down1 = self.down(conv1)
        conv2 = self.layer2(down1)
        down2 = self.down(conv2)
        conv3 = self.layer3(down2)
        down3 = self.down(conv3)
        conv4 = self.layer4(down3)
        down4 = self.down(conv4)
        conv5 = self.layer5(down4)
        up1 = self.up1(conv5, conv4)
        conv6 = self.layer6(up1)
        up2 = self.up2(conv6, conv3)
        conv7 = self.layer7(up2)
        up3 = self.up3(conv7, conv2)
        conv8 = self.layer8(up3)
        up4 = self.up4(conv8, conv1)
        conv9 = self.layer9(up4)
        out = self.layer10(conv9)
    
        self.out = out.permute(0,2,3,1)

        
    def spot_expression(self, is_train):
        
        #self.sc_trans = self.sc_data        
        self.sc_trans = self.sc_data * (F.softplus(self.sc_drop))
        self.sc_trans = self.sc_trans / (torch.sum(self.sc_trans, dim=1)[:,None] + 1e-6) * self.sc_data.shape[1]
        #self.sc_trans = self.sc_data

        
        
        self.mask = self.mask.permute(0,2,3,1)
        self.mask = self.mask.unsqueeze(-1)

        self.out = self.layer_norm_pixel(self.out2out(self.out))
        self.expression = torch.sum(self.mask * self.out.unsqueeze(3).repeat(1,1,1,self.max_cell,1), dim=[1,2]).reshape([-1,200]) / (torch.sum(self.mask, dim=[1,2,4])[:,None] + 1e-3).reshape(-1,1)
        #self.expression = torch.sum(self.mask * self.out.unsqueeze(3).repeat(1,1,1,self.max_cell,1), dim=[1,2]).reshape([-1,200]) / (self.ave_mask.reshape(-1,1).repeat(1,self.max_cell).reshape(-1)[:,None] + 1e-6)

        #print(self.ave_mask, self.ave_mask.shape)

        
        attention_weight = torch.sum(torch.multiply(torch.flatten(self.out.permute(0,3,1,2).unsqueeze(1).repeat(1,self.max_cell,1,1,1), start_dim=0, end_dim=1), self.expression.unsqueeze(-1).unsqueeze(-1)).permute(0,2,3,1),dim=-1) / np.sqrt(200)
        attention_weight = torch.softmax(attention_weight.reshape(attention_weight.shape[0],-1),dim=1).reshape(attention_weight.shape) 
        #print(attention_weight.shape, self.expression.shape, self.out.shape)
        self.expression = self.expression + torch.sum(attention_weight.unsqueeze(-1) * torch.flatten(self.out.unsqueeze(1).repeat(1,self.max_cell,1,1,1), start_dim=0, end_dim=1), dim=[1,2])




        if is_train:
            self.sc_exp = self.dropout(torch.relu(self.sc2feature1(self.layer_norm_scExp(self.sc_data))))
            self.sc_exp = self.dropout(torch.relu(self.sc2feature2(self.sc_exp)))
            self.sc_exp = self.dropout(torch.tanh(self.sc2feature3(self.sc_exp)))
            if self.mode == "mapping":
                #self.abundance = F.gumbel_softmax(torch.matmul(self.expression, self.sc_exp.T), tau=0.1, hard=False, dim=1)
                self.abundance = F.gumbel_softmax(torch.matmul(self.expression, self.sc_exp.T) / np.sqrt(self.nCount_var), tau=0.1, hard=False, dim=1)
                #self.abundance = torch.softmax(torch.matmul(self.expression, self.sc_exp.T) / np.sqrt(self.nCount_var), dim=1)
                #self.abundance = F.softplut(torch.matmul(self.expression, self.sc_exp.T))
                #self.abundance = self.abundance / (torch.sum(self.abundance, dim=1)[:,None] + 1e-7)
            else:
                self.abundance = F.softplus(torch.matmul(self.dropout(self.expression), self.sc_exp.T))
                self.abundance = self.abundance / (torch.sum(self.abundance, dim=1)[:,None] + 1e-7)
            
        else:
            self.sc_exp = torch.relu(self.sc2feature1(self.layer_norm_scExp(self.sc_data)))
            self.sc_exp = torch.relu(self.sc2feature2(self.sc_exp))
            self.sc_exp = torch.tanh(self.sc2feature3(self.sc_exp))
            if self.mode == "mapping":
                #self.abundance = torch.softmax(torch.matmul(self.expression, self.sc_exp.T), dim=1)
                self.abundance = torch.softmax(torch.matmul(self.expression, self.sc_exp.T) / np.sqrt(self.nCount_var), dim=1) 
            else:
                self.abundance = F.softplus(torch.matmul(self.expression, self.sc_exp.T))
                self.abundance = self.abundance / (torch.sum(self.abundance, dim=1)[:,None] + 1e-7)
            
       
        self.cell_type_abundance = torch.matmul(self.abundance, self.anno_class)
        #self.prob = self.cell_type_abundance / (torch.sum(self.cell_type_abundance, dim=1)[:,None] + 1e-7)
        #self.entropy = - torch.sum(self.prob * torch.log(self.prob + 1e-7), dim=1) / np.sqrt(self.anno_class.shape[1])
        
        #print(self.cell_type_abundance[0:3,:])        
        ## entropy 1 decrease cell used
        #self.prob = self.abundance / (torch.sum(self.abundance, dim=1)[:,None] + 1e-7)
        #self.entropy1 = - torch.sum(self.prob * torch.log(self.prob + 1e-7), dim=1) / np.sqrt(self.nCount_obs)
        self.prob = self.cell_type_abundance / (torch.sum(self.cell_type_abundance, dim=1)[:,None] + 1e-7)
        self.entropy1 = - torch.sum(self.prob * torch.log(self.prob + 1e-7), dim=1) / np.sqrt(self.anno_class.shape[1])

        ## entropy 2 improve cell type used
        self.cell_type_abundance2 = torch.sum(self.cell_type_abundance, dim=0)
        self.prob2 = self.cell_type_abundance2 / (torch.sum(self.cell_type_abundance2) + 1e-7)
        self.entropy2 = - torch.sum(self.prob2 * torch.log(self.prob2 + 1e-7)) / np.sqrt(self.anno_class.shape[1])
    
    
        #if self.mode == "mapping" and not is_train:
        if self.mode == "mapping":
            index = torch.argmax(self.abundance, dim=1)
            self.abundance = self.abundance * 0
            self.abundance[range(len(index)), index] = 1
        
        self.sc_predict = torch.matmul(self.abundance, self.sc_trans)
        self.sc_predict = self.sc_predict / (torch.sum(self.sc_predict, dim=-1) + 1e-6)[:,None] * self.nCount_var

        #print(self.sc_predict, self.sc_predict.shape)        
        self.sc_predict = self.sc_predict.reshape([-1,self.max_cell,self.nCount_var])
        self.sc_predict = torch.sum(self.sc_predict, dim=1)
        
        
        
        
        self.sc_predict = self.sc_predict / (torch.sum(self.sc_predict, dim=1)[:,None] + 1e-7) * self.nCount_var
        self.exp = self.exp / (torch.sum(self.exp, dim=1)[:,None] + 1e-7) * self.nCount_var
        
        self.trans_similarity = torch.cosine_similarity(self.sc_trans, self.sc_data, dim=1)
        self.expression_similarity = torch.cosine_similarity(self.exp, self.sc_predict, dim=1)
        #self.expression_similarity = torch.cosine_similarity(torch.log(self.exp+1), torch.log(self.sc_predict+1), dim=1)
        #self.expression_similarity2 = torch.cosine_similarity(torch.log(self.exp.T+1), torch.log(self.sc_predict.T+1), dim=1)
        #self.expression_similarity = torch.cosine_similarity(self.exp, self.sc_predict, dim=1)
        
        #self.loss_st_sim = 1 - torch.mean(self.expression_similarity) + 1 - torch.mean(self.expression_similarity2)
        self.loss_st_sim = 1 - torch.mean(self.expression_similarity)
        self.loss_trans_sim = (torch.matmul(torch.sum(self.abundance, dim=0), 1 - self.trans_similarity) / torch.sum(self.abundance))

        
        self.loss_entropy1 = torch.mean(self.entropy1)
        self.loss_entropy2 = 1 - torch.mean(self.entropy2)
        self.loss = self.loss_st_sim + self.loss_trans_sim + 0.0 * self.loss_entropy1 + 0.05 * self.loss_entropy2
         
        
           
        

    def training_step(self, batch, batch_idx):
        
        is_train = True
        
        patch, mask, exp, index = batch
        
        self.patch = patch
        self.mask = mask
        #self.ave_mask = ave_mask
        self.exp = exp
        
        self.forward(self.patch)
        self.spot_expression(is_train)
      
      
        self.total_loss.append(self.loss.item())
        self.total_loss_st_sim.append(self.loss_st_sim.item())
        self.total_loss_trans_sim.append(self.loss_trans_sim.item())
        self.total_loss_entropy1.append(self.loss_entropy1.item())
        self.total_loss_entropy2.append(self.loss_entropy2.item())      
        
     
        return self.loss
        
    
    def validation_step(self, batch, batch_idx):
    
        is_train = False
        
        patch, mask, exp, index = batch
        
        self.patch = patch
        self.mask = mask
        self.exp = exp
       
        
        self.forward(self.patch)
        self.spot_expression(is_train)
        
        self.real_predict.append(self.expression_similarity.detach().cpu())
        #self.real_real.append(torch.cosine_similarity(self.exp, self.exp[torch.randperm(self.exp.shape[0]),:], dim=1).detach().cpu())
        #self.predict_predict.append(torch.cosine_similarity(self.sc_predict, self.sc_predict[torch.randperm(self.sc_predict.shape[0]),:], dim=1).detach().cpu())
        self.real_real.append(torch.cosine_similarity(torch.log(self.exp+1), torch.log(self.exp[torch.randperm(self.exp.shape[0]),:]+1), dim=1).detach().cpu())
        self.predict_predict.append(torch.cosine_similarity(torch.log(self.sc_predict+1), torch.log(self.sc_predict[torch.randperm(self.sc_predict.shape[0]),:]+1), dim=1).detach().cpu())
       
 
        return self.loss
    
    
    def validation_epoch_end(self, outputs):
        
        self.epoch += 1
        
        print("Total_loss:",np.mean(self.total_loss),"total_loss_st_sim:",np.mean(self.total_loss_st_sim),
              "total_loss_trans_sim:",np.mean(self.total_loss_trans_sim),"total_loss_entropy1:",np.mean(self.total_loss_entropy1),"total_loss_entropy2:",np.mean(self.total_loss_entropy2))
        
        self.total_loss = []
        self.total_loss_st_sim = []
        self.total_loss_trans_sim = []
        self.total_loss_entropy1 = []
        self.total_loss_entropy2 = []
        
        self.real_predict = torch.concat(self.real_predict)
        self.predict_predict = torch.concat(self.predict_predict)
        self.real_real = torch.concat(self.real_real)
        
        print("real vs predict", torch.mean(self.real_predict).item())
        print("real vs real", torch.mean(self.real_real).item())
        print("predict vs predict", torch.mean(self.predict_predict).item())
        print("----------------------")
        
        self.real_predict = []
        self.predict_predict = []
        self.real_real = [] 

    

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return self.optimizer








     




