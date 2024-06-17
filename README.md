# RedeHist
Spatial Transcriptomics Prediction at Single-cell Resolution on Histology Images using RedeHist

![figure1](https://github.com/Roshan1992/RedeHist/assets/11591480/fe053163-b09b-42fe-83c7-2abddd4f4f8f)



## Overview

RedeHist is an automatic tool for spatial transcriptomics (ST) prediction at single-cell resolution on histology images. This approach employs a deep neural network integrated with nuclei segmentation results to predict transcriptomic profiles at single-cell resolution from histology images. RedeHist takes histology images, ST data, and scRNA-seq references as inputs, then generates outputs consisting of single cells identified on the images along with their whole transcriptomic expression profiles, spatial coordinates, and annotations. RedeHist is a python package written in Python 3.9 and pytorch 1.12. It allows GPU to accelerate computational efficiency.


## Installation

[1] Install <a href="https://www.anaconda.com/" target="_blank">Anaconda</a> if not already available

[2] Clone this repository:
```
    git clone https://github.com/Roshan1992/RedeHist.git
```

[3] Change to RedeHist directory:
```
    cd RedeHist
```

[4] Create a conda environment with the required dependencies:
```
    conda env create -f environment.yml
```

[5] Activate the RedeHist_env environment you just created:
```
    conda activate RedeHist_env
```

[6] Install RedeHist:
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

### [1] Download example data
Click <a href="https://drive.google.com/file/d/1Y93zWhKSbLqNg31i-pwlew7-nQNHOJIJ/view?usp=drive_link" target="_blank">here</a> to download example data for RedeHist. Then, place the downloaded file (example_data_for_RedeHist.zip) into the RedeHist directory and unzip it.
```
    unzip example_data_for_RedeHist.zip
```

### [2] Run RedeHist for example datasets

__[1] To implement RedeHist for Imaging-based ST:__ please run

```
    python -u example_train_ImagingBased.py
    python -u example_predict_ImagingBased.py
```
for training and prediction respectively.

__[2]To implement RedeHist for Sequencing-based ST:__ please run

```
    python -u example_train_SequencingBased.py
    python -u example_predict_SequencingBased.py
```
for training and prediction respectively.

### [3] Output
The output of RedeHist is a h5ad file that includes predicted cells with their whole transcriptomic expression profiles, spatial coordinates, and cell annotations.


