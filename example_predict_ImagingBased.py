# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 09:48:02 2023

@author: 32792
"""

import os
import RedeHist
import numpy as np
import scanpy as sc
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

device = "cuda"
max_cell = 1
batch_size = 32
resolution = 128
sample_name = "HBC_Xenium"
work_dir = "example_Xenium"
sc_path = "example_data/sc_data.FFPE.anno.h5ad"
st_path = os.path.join(work_dir, sample_name + ".st_data.h5ad")
model_path = os.path.join(work_dir, "epoch=49.ckpt")
output_expression_path = os.path.join(work_dir, sample_name + ".predict.h5ad" )
test_image_path = "example_data/Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.tif"



img = Image.open(test_image_path)
img = np.asarray(img)[:,:,0:3]
nuclei_masks = RedeHist.cell_segmentation(img, use_gpu=True)
all_image, all_masks, use_center, index2cell = RedeHist.preprocess_prediction(img, nuclei_masks, resolution = resolution)



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



model = RedeHist.RedeHist_model.load_from_checkpoint(model_path, sc_data = sc_data_use, max_cell = max_cell, device = device, strict=False)
model.to(device)
 


RedeHist.prediction(model, 
                    all_image, 
                    all_masks, 
                    use_center, 
                    index2cell, 
                    sc_data, 
                    sc_data_use, 
                    output_expression_path,
                    batch_size=batch_size, 
                    whole_transcriptome=True,
                    device=device)

















