# RedeHist
Spatial Transcriptomics Prediction at Single-cell Resolution on Histology Images using RedeHist

![workflow](https://user-images.githubusercontent.com/11591480/236590851-2abd1da4-8900-42a7-813e-35ec816c5129.png)



## Overview

RedeFISH is an automatic tool for cell alignment in imaging-based spatial transcriptomics (ST) and scRNA-seq data through deep reinforcement learning. This method aims to identify functional-defined cells in ST data that exhibit the highest degree of expression similarity with cells in scRNA-seq data. Through the incorporation of scRNA-seq data, this method additionally undertakes the task of inferring whole-transcriptome expression profiles for the aforementioned identified cells. RedeFISH is a python package written in Python 3.9 and pytorch 1.12. It allows GPU to accelerate computational efficiency.


## Installation

[1] Install <a href="https://www.anaconda.com/" target="_blank">Anaconda</a> if not already available

[2] Clone this repository:
```
    git clone https://github.com/Roshan1992/RedeFISH.git
```

[3] Change to RedeFISH directory:
```
    cd RedeHIST
```

[4] Create a conda environment with the required dependencies:
```
    conda env create -f environment.yml
```

[5] Activate the RedeHist_env environment you just created:
```
    conda activate RedeHist_env
```

[6] Install RedeHIST:
```
    pip install .
```

[7] Install pytorch:

If GPU available (https://pytorch.org/get-started/previous-versions/):
```
    pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```
If GPU not available:
```
    pip install torch==1.12.1+cpu torchvision==0.13.1+cpu torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cpu
```

## Quick Start

### Download example data


### Run RedeHist

RedeFISH requires 2 file as input:

__[1] A csv file for single-cell ST data:__ This file must includes at least 3 columns, namely __x__, __y__ and corresponding __gene__.

![image](https://user-images.githubusercontent.com/11591480/236604144-21a769c2-398b-40e2-9dc7-084d7630241d.png)

__[2] An Anndata h5ad file for scRNA-seq data:__ This file must includes expression matrix and cell type annotation.

![image](https://user-images.githubusercontent.com/11591480/236605176-6551c703-e19b-42f0-9c43-4022e41b7eb4.png)

### Output

See <a href="https://github.com/Roshan1992/Redesics/blob/main/example.ipynb" target="_blank">example</a> for implementing Redesics on imaging-based single-cell ST platforms.

See <a href="https://github.com/Roshan1992/Redesics/blob/main/example_for_Stereo_seq.ipynb" target="_blank">example_for_Stereo_seq</a> for implementing Redesics on Stereo-seq platforms.


