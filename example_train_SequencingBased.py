# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 08:20:56 2023

@author: 32792
"""


import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import scanpy as sc

import RedeHist
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


train_resolution = 192
max_cell = 20

work_dir = "example_Visium"
sample_name = "HBC_Visium"

if __name__ == '__main__':
    
      
    image_path = "example_data/CytAssist_FFPE_Human_Breast_Cancer_tissue_image.tif"
    st_data = sc.read("example_data/st_data.HBC_visium.h5ad")
    
    img = Image.open(image_path)
    img = np.asarray(img)[:,:,0:3]
    
    nuclei_masks = RedeHist.cell_segmentation(img, use_gpu=True)
    RedeHist.preprocess_sequencing_based_ST(img, st_data, nuclei_masks, sample_name, work_dir, max_resolution=256)
    
   
    img_path = os.path.join(work_dir, sample_name + ".training_images.npy")
    mask_path = os.path.join(work_dir, sample_name + ".training_masks.npy")
    sc_path = "example_data/sc_data.FFPE.anno.h5ad"
    st_path = "example_data/st_data.HBC_visium.h5ad"
    align_path = os.path.join(work_dir, sample_name + ".spot_align_cell.csv")
    
    
    
    
    sc_data = sc.read(sc_path)
    st_data = sc.read(st_path)
    sc_data.var_names_make_unique()
    sc_data.obs_names_make_unique()
    st_data.var_names_make_unique()
    st_data.obs_names_make_unique()
    inter_gene = np.intersect1d(sc_data.var_names, st_data.var_names)
    sc_data_use = sc_data[:, inter_gene]
    st_data_use = st_data[:, inter_gene]
    sc.pp.filter_cells(sc_data_use, min_genes=10)
    
    
    
    
    df, gene2score = RedeHist.gene_weight_ROGUE(sc_data_use)
    number_gene = 2000
    df = df.iloc[0:number_gene]
    sc_data_use = sc_data_use[:, df['use_gene']]
    st_data_use = st_data_use[:, df['use_gene']]

    df.to_csv(os.path.join(work_dir, "gene_list.txt"), sep='\t')

    sc.pp.normalize_per_cell(sc_data_use, counts_per_cell_after=number_gene)
    sc.pp.normalize_per_cell(st_data_use, counts_per_cell_after=number_gene)
    
    
    np.random.seed(0)
    index = list(range(len(st_data)))
    np.random.shuffle(index)
    
    train_dataset = RedeHist.data_utils(img_path, mask_path, st_data_use, align_path, 
                                        max_cell=max_cell, train_resolution=train_resolution, index=index[0:])
    val_dataset = RedeHist.data_utils(img_path, mask_path, st_data_use, align_path,
                                      max_cell=max_cell, train_resolution=train_resolution, index= index[0:500])  
    train_loader = DataLoader(train_dataset, batch_size=8, num_workers=8, shuffle=True)    
    val_loader = DataLoader(val_dataset, batch_size=8, num_workers=8, shuffle=False)   
    
    
    
    #model = RedeHist.RedeHist_model(sc_data_use, max_cell, "cuda")
    model = RedeHist.RedeHist_model(sc_data_use, max_cell, device = "cuda")
    trainer = pl.Trainer(gpus=[3,4], max_epochs=50, 
                         callbacks=pl.callbacks.ModelCheckpoint(  
                            dirpath=work_dir,  
                            filename='{epoch}',  
                            save_top_k=1
                            ), 
                         strategy="dp")
    trainer.fit(model, train_loader, val_loader)   





    


#adata[adata[:,"CD68"].X > 0, :]


