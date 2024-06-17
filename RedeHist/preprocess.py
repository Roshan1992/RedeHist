# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 10:18:37 2023

@author: 32792
"""


import torch
import numpy as np
import pandas as pd
from cellpose import io, models

from pandas import DataFrame
import os, re
import cv2 as cv
import scanpy as sc

from scipy.spatial import KDTree
from sklearn import neighbors
from scipy import stats
from scipy.interpolate import UnivariateSpline




def output_intermediate(output_path):
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
        
        
        
def gene_weight_ROGUE(sc_data):

    expr = sc_data.to_df()
    expr = expr / np.asarray(np.sum(expr, axis=1))[:,None] * expr.shape[1]
    tmp = np.log(expr + 1)
    entropy = np.asarray(np.mean(tmp, axis=0))
    mean_expr = np.asarray(np.log(np.mean(expr, axis=0) + 1))


    prd, spl = process_smooth_spline(mean_expr, entropy)
    distance = prd - entropy
    pv = stats.norm.cdf(x=distance, loc=np.mean(distance), scale=np.std(distance))

    entropy2 = entropy[pv <= 0.9]
    mean_expr2 = mean_expr[pv <= 0.9]

    prd, spl = process_smooth_spline(mean_expr2, entropy2)
    prd = spl(mean_expr)
    distance = prd - entropy

    xs = np.linspace(0, max(mean_expr), 500)

    use_gene_score = stats.norm.cdf(x=distance, loc=np.mean(distance), scale=np.std(distance))

    df = pd.DataFrame()
    df['use_gene_score'] = use_gene_score
    df['use_gene'] = sc_data.var_names
    df['mean_expression'] = np.mean(np.asarray(sc_data.to_df()),axis=0)
    df = df.sort_values(by=['use_gene_score'], ascending=False)

    gene2score = {}
    for a,b in zip(sc_data.var_names, use_gene_score):
        gene2score[a] = b

    return(df, gene2score)
        
        
        
def process_smooth_spline(x, y):


    index = np.argsort(x)
    x2 = np.sort(x)
    y2 = y[index]

    spl = UnivariateSpline(x2, y2)

    prd = spl(x)

    return(prd, spl)
        
        
        
        
        
def cell_segmentation(img,
                      min_size = 15,
                      flow_threshold = 0.8,
                      diameter=None, 
                      chan = [1,0],
                      use_gpu=False):
    
    print("Perform nuclei segmentation ...")
    model = models.Cellpose(gpu=use_gpu, model_type='nuclei')
    # define CHANNELS to run segementation on
    # grayscale=0, R=1, G=2, B=3
    # channels = [cytoplasm, nucleus]
    
    # segment image
    nuclei_masks, flows, styles, diams = model.eval(img, flow_threshold=flow_threshold, min_size=min_size, diameter=diameter, channels=chan, invert=True)
    #nuclei_masks, flows, styles, diams = model.eval(img, flow_threshold=flow_threshold, min_size=min_size, diameter=diameter, invert=True)
 
    print("Max cell masks:", np.max(nuclei_masks))
    
    return(nuclei_masks)
    

def preprocess_sequencing_based_ST(img,
                                   st_data,
                                   nuclei_masks, 
                                   sample_name,
                                   work_dir,
                                   spot_radius = 128, 
                                   max_resolution = 512,
                                   cell_mRNA_cutoff = 20,
                                   ):    
    
    res = max_resolution
    
    a = torch.tensor(np.int32(nuclei_masks))
    idx = torch.nonzero(a).T
    data = a[idx[0],idx[1]]
    
    data_index = torch.stack([data-1, torch.zeros(len(data))], dim=0)
    x_sum = torch.sparse_coo_tensor(data_index, idx[0,:], (torch.max(data), 1))
    y_sum = torch.sparse_coo_tensor(data_index, idx[1,:], (torch.max(data), 1))
    x_count = torch.sparse_coo_tensor(data_index, torch.ones(len(data)), (torch.max(data), 1))
    y_count = torch.sparse_coo_tensor(data_index, torch.ones(len(data)), (torch.max(data), 1))
    
    x_center = x_sum.to_dense() / x_count.to_dense()
    y_center = y_sum.to_dense() / y_count.to_dense()
    
    use_center = torch.stack([torch.tensor(range(1,torch.max(data)+1,1)), x_center.reshape(-1), y_center.reshape(-1)]).T.numpy()


    st_data.obsm['spatial'] = np.float32(st_data.obsm['spatial'])
    left = np.int32(st_data.obsm['spatial'][:,1] - res / 2)
    right = left + res
    bottom = np.int32(st_data.obsm['spatial'][:,0] - res / 2)
    up = bottom + res
    
    
    all_image = []
    all_masks = []
    for i in range(len(st_data)):
        all_image.append(img[left[i]:right[i], bottom[i]:up[i], :].copy())
        all_masks.append(nuclei_masks[left[i]:right[i], bottom[i]:up[i]].copy())

    tree = neighbors.KDTree(use_center[:,[2,1]], leaf_size=2)          
    nearby = tree.query_radius(np.int32(st_data.obsm['spatial']), spot_radius) 
        
    index2cell = dict()
    for i in range(len(nearby)):
        index2cell[str(int(use_center[i,0]))] = [str(x+1) for x in nearby[i]]
        
    all_image = np.stack(all_image)
    all_masks = np.stack(all_masks)
    
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    np.save(os.path.join(work_dir, sample_name + ".training_images.npy"), all_image)
    np.save(os.path.join(work_dir, sample_name + ".training_masks.npy"), all_masks)
    with open(os.path.join(work_dir, sample_name + ".spot_align_cell.csv"), "w", encoding='utf8') as w:
        for each in index2cell:
            w.write(",".join(index2cell[each]) + '\n')
        

def preprocess_imaging_based_ST(img,
                                transcripts, 
                                nuclei_masks, 
                                sample_name,
                                work_dir,
                                scale = 15, 
                                max_resolution = 128,
                                cell_mRNA_cutoff = 20
                                ):
    
    
    
    res = max_resolution
    
    gene = list(transcripts['gene'])
    gene_list = list(set(gene))
    gene_list.sort()
    gene2index = dict()
    for i in range(len(gene_list)):
        gene2index[gene_list[i]] = i
    trans_points = np.asarray([transcripts['x'], transcripts['y']]).T
    

    a = torch.tensor(np.int32(nuclei_masks))
    idx = torch.nonzero(a).T
    data = a[idx[0],idx[1]]
    
    data_index = torch.stack([data-1, torch.zeros(len(data))], dim=0)
    x_sum = torch.sparse_coo_tensor(data_index, idx[0,:], (torch.max(data), 1))
    y_sum = torch.sparse_coo_tensor(data_index, idx[1,:], (torch.max(data), 1))
    x_count = torch.sparse_coo_tensor(data_index, torch.ones(len(data)), (torch.max(data), 1))
    y_count = torch.sparse_coo_tensor(data_index, torch.ones(len(data)), (torch.max(data), 1))
    
    x_center = x_sum.to_dense() / x_count.to_dense()
    y_center = y_sum.to_dense() / y_count.to_dense()
    
    use_center = torch.stack([torch.tensor(range(1,torch.max(data)+1,1)), x_center.reshape(-1), y_center.reshape(-1)]).T.numpy()
    
    
    tree = KDTree(data=idx.T.numpy())
    distance, cell_index = tree.query(trans_points, k=1, workers=16)   
    
    mRNA_index = torch.stack([data[cell_index]-1, torch.tensor([gene2index[x] for x in gene])])
    mRNA_index = mRNA_index[:, distance <= scale]
    mRNA_sparse = torch.sparse_coo_tensor(mRNA_index, torch.ones(mRNA_index.shape[1]), (torch.max(data), len(gene2index)))
    cell_expression = mRNA_sparse.to_dense().type(torch.int)
    
        
    st_data = sc.AnnData(cell_expression.numpy())
    st_data.var_names = gene_list
    st_data.obs_names = [str(int(x)) for x in use_center[:,0]]
    st_data.obsm['spatial'] = use_center[:,1:3]
    sc.pp.filter_cells(st_data, min_counts=cell_mRNA_cutoff)
    
    
    
    left = np.int32(st_data.obsm['spatial'][:,0] - res / 2)
    right = left + res
    bottom = np.int32(st_data.obsm['spatial'][:,1] - res / 2)
    up = bottom + res
    use_cell = (left >= 0) & (right < img.shape[0]) & (bottom >= 0) & (up < img.shape[1])
    
    st_data = st_data[use_cell, :]
    left = np.int32(st_data.obsm['spatial'][:,0] - res / 2)
    right = left + res
    bottom = np.int32(st_data.obsm['spatial'][:,1] - res / 2)
    up = bottom + res
    
    
    
    all_image = []
    all_masks = []
    index2cell = dict()
    for i in range(len(st_data)):
        all_image.append(img[left[i]:right[i], bottom[i]:up[i], :].copy())
        all_masks.append(nuclei_masks[left[i]:right[i], bottom[i]:up[i]].copy())
        cell_name = str(int(float(st_data.obs_names[i])))
        index2cell[cell_name] = [cell_name]

    all_image = np.stack(all_image)
    all_masks = np.stack(all_masks)
    
    
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
     
    sc.write(os.path.join(work_dir, sample_name + ".st_data.h5ad"), st_data)
    np.save(os.path.join(work_dir, sample_name +  ".training_images.npy"), all_image)
    np.save(os.path.join(work_dir, sample_name + ".training_masks.npy"), all_masks)
    with open(os.path.join(work_dir, sample_name + ".spot_align_cell.csv"), "w", encoding='utf8') as w:
        for each in index2cell:
            w.write(",".join(index2cell[each]) + '\n')



def preprocess_prediction(img,
                          nuclei_masks,   
                          resolution = 128
                          ):
    
    a = torch.tensor(np.int32(nuclei_masks))
    idx = torch.nonzero(a).T
    data = a[idx[0],idx[1]]
    
    data_index = torch.stack([data-1, torch.zeros(len(data))], dim=0)
    x_sum = torch.sparse_coo_tensor(data_index, idx[0,:], (torch.max(data), 1))
    y_sum = torch.sparse_coo_tensor(data_index, idx[1,:], (torch.max(data), 1))
    x_count = torch.sparse_coo_tensor(data_index, torch.ones(len(data)), (torch.max(data), 1))
    y_count = torch.sparse_coo_tensor(data_index, torch.ones(len(data)), (torch.max(data), 1))
    
    x_center = x_sum.to_dense() / x_count.to_dense()
    y_center = y_sum.to_dense() / y_count.to_dense()
    
    use_center = torch.stack([torch.tensor(range(1,torch.max(data)+1,1)), x_center.reshape(-1), y_center.reshape(-1)]).T.numpy()
    
    
    left = np.int32(use_center[:,1] - resolution / 2)
    left[left < 0] = 0
    right = left + resolution
    bottom = np.int32(use_center[:,2] - resolution / 2)
    bottom[bottom < 0] = 0
    up = bottom + resolution
    

    right[right > img.shape[0] - 1] = img.shape[0] - 1
    up[up > img.shape[1] - 1] = img.shape[1] - 1
    left = right - resolution
    bottom = up - resolution
    
    
    
    
    all_image = []
    all_masks = []
    index2cell = dict()
    for i in range(len(use_center)):
        all_image.append(img[left[i]:right[i], bottom[i]:up[i], :].copy())
        nuclei_masks_temp = nuclei_masks[left[i]:right[i], bottom[i]:up[i]].copy()
        nuclei_masks_temp[nuclei_masks_temp != use_center[i,0]] = 0
        all_masks.append(nuclei_masks_temp)
        cell_name = str(int(float(use_center[i,0])))
        index2cell[cell_name] = [cell_name]
    
    all_image = np.stack(all_image)
    all_masks = np.stack(all_masks)
 
    return(all_image, all_masks, use_center, index2cell)
    



    
    
    
    
    
    
    
    
    
    
    
    
