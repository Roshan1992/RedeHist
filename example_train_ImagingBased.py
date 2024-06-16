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



train_resolution = 128
max_cell = 1

work_dir = "example_Xenium"
sample_name = "HBC_Xenium"

if __name__ == '__main__':
    
    train_image_path = "example_data/Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.tif"
    transcripts = pd.read_csv("example_data/transcripts_HE.csv")
     
    img = Image.open(train_image_path)
    img = np.asarray(img)[:,:,0:3]
   
    nuclei_masks = RedeHist.cell_segmentation(img, use_gpu=True)
    RedeHist.preprocess_imaging_based_ST(img, transcripts, nuclei_masks, sample_name, work_dir)
    
    img_path = os.path.join(work_dir, sample_name + ".training_images.npy")
    mask_path = os.path.join(work_dir, sample_name + ".training_masks.npy")
    st_path = os.path.join(work_dir, sample_name + ".st_data.h5ad")
    sc_path = "example_data/sc_data.FFPE.anno.h5ad"
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
    
    
    np.random.seed(0)
    index = list(range(len(st_data)))
    np.random.shuffle(index)
    
    
    train_dataset = RedeHist.data_utils(img_path, mask_path, st_data_use, align_path, 
                                        max_cell=max_cell, train_resolution=train_resolution, index=index[0:])
    val_dataset = RedeHist.data_utils(img_path, mask_path, st_data_use, align_path,
                                      max_cell=max_cell, train_resolution=train_resolution, index= index[0:5000])  
    train_loader = DataLoader(train_dataset, batch_size=64, num_workers=8, shuffle=True)    
    val_loader = DataLoader(val_dataset, batch_size=64, num_workers=8, shuffle=False)   
    
    
    
    model = RedeHist.RedeHist_model(sc_data_use, max_cell, "cuda")
    trainer = pl.Trainer(gpus=[0,2], 
                         max_epochs=50, 
                         callbacks=pl.callbacks.ModelCheckpoint(  
                            dirpath=work_dir,  
                            filename='{epoch}',  
                            save_top_k=1
                            ),  
                         strategy="dp")
    trainer.fit(model, train_loader, val_loader)   
    
    
    
    

