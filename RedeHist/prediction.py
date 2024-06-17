# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 11:12:22 2023

@author: 32792
"""


import torch
import numpy as np
import scanpy as sc


def prediction(model, 
               all_image, 
               all_masks, 
               cell_center, 
               index2cell, 
               sc_data, 
               sc_data_use, 
               output_expression_path,
               batch_size=1, 
               whole_transcriptome=True,
               device='cpu'):
    
    
    
    sc_data = sc_data[sc_data_use.obs_names, :]
    ratio = np.float32(sc_data_use.n_vars / (np.sum(np.asarray(sc_data_use.to_df()),axis=1) + 1e-7))
    if whole_transcriptome:
        sc_data_numpy = torch.tensor(np.float32(sc_data.to_df()) * ratio[:,None]).to(device)
    else:
        sc_data_numpy = 0
    
    align = list(index2cell.keys())
    cell_type = list(sc_data.obs['annotation'])
    
    CT2index = dict()
    for i in range(len(cell_type)):
        if not cell_type[i] in CT2index:
            CT2index[cell_type[i]] = []
        CT2index[cell_type[i]].append(i)
    for each in CT2index:
        CT2index[each] = np.asarray(CT2index[each])
    
    
    if whole_transcriptome:
        gene_name = sc_data.var_names
        predict_expression = np.zeros([cell_center.shape[0], len(gene_name)])
    else:
        gene_name = sc_data_use.var_names
        predict_expression = np.zeros([cell_center.shape[0], len(gene_name)])
    
    
    exp = torch.zeros([batch_size, len(sc_data_use.var_names)]).float().to(device)
    model.exp = exp.float().to(device)
    
    
    index = 0
    batch_images = []
    batch_mask = []
    CT = []
    CT_keys = list(CT2index.keys())
    
    for i in range(cell_center.shape[0]):
    
        if i % 10000 == 0:
            print(i)
    
        image_this = all_image[i].copy()
        masks_this = all_masks[i].copy()
        masks_this[masks_this == int(align[i])] = 10000000
        masks_this[masks_this < 9999999] = 0
        masks_this[masks_this > 9999999] = 1
        
    
        batch_images.append(image_this.copy())
        batch_mask.append(masks_this.copy())
    
    
        if len(batch_images) == batch_size or i == cell_center.shape[0] - 1:
            batch_images = np.stack(batch_images)
            batch_images = torch.from_numpy(batch_images)
            batch_images = batch_images / 255
            batch_images = batch_images.permute(0,3,1,2)
    
            batch_mask = np.float32(np.stack(batch_mask))
            batch_mask = torch.from_numpy(batch_mask)
            batch_mask = torch.stack([batch_mask for x in range(1)]).permute(1,0,2,3)
    
            model.patch = batch_images.float().to(device)
            model.mask = batch_mask.float().to(device)
            model.ave_mask = torch.sum(model.mask, dim=[1,2,3])
            model.exp = exp[0:batch_images.shape[0],:].float().to(device)
    
            model.forward(model.patch)
            model.spot_expression(False)
    
            model.abundance = model.abundance / (torch.sum(model.abundance, dim=1)[:,None] + 1e-7)
            CT_values = np.zeros([model.abundance.shape[0], len(CT_keys)])
            for j in range(len(CT_keys)):
                CT_values[:,j] = torch.sum(model.abundance[:, CT2index[CT_keys[j]]], dim=1).detach().cpu().numpy()
            CT_this = np.argmax(CT_values, axis=1)
            #print(model.abundance.detach().dtype, sc_data_numpy.dtype) 
            if whole_transcriptome:
                temp = torch.matmul(model.abundance.detach(), sc_data_numpy).cpu().numpy()
            else:
                temp = model.sc_predict.detach().cpu().numpy()
    
            for j in range(len(temp)):
                predict_expression[index, :] = temp[j,:].copy()
                CT.append(CT_keys[CT_this[j]])
                index += 1
    
            batch_images = []
            batch_mask = [] 
            
            
    sc_out = sc.AnnData(predict_expression)
    sc_out.var_names = gene_name
    sc_out.obs['annotation'] = CT
    sc_out.obsm['spatial'] = np.asarray(cell_center[:,1:3])
    sc.write(output_expression_path, sc_out)
    
















