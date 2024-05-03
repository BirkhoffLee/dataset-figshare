# Figshare Brain Tumor MRI Dataset

This brain tumor dataset containing 3064 T1-weighted contrast-inhanced images
from 233 patients with three kinds of brain tumor: meningioma (708 slices), 
glioma (1426 slices), and pituitary tumor (930 slices). The images are 512x512 pixels.

Original dataset was published here: https://doi.org/10.6084/m9.figshare.1512427.v5

This dataset utilises [the converted version](https://www.kaggle.com/datasets/denizkavi1/brain-tumor/data) of the original dataset. I host the zip file on my own server to make it easier to download, and made a TFDS dataset from it, which is this repo.

Note that I have not yet implemented the test scripts; please contribute if you can.

## Original Description

> This brain tumor dataset containing 3064 T1-weighted contrast-inhanced images
> from 233 patients with three kinds of brain tumor: meningioma (708 slices), 
> glioma (1426 slices), and pituitary tumor (930 slices). Due to the file size
> limit of repository, we split the whole dataset into 4 subsets, and achive 
> them in 4 .zip files with each .zip file containing 766 slices.The 5-fold
> cross-validation indices are also provided.
> 
> -----
> This data is organized in matlab data format (.mat file). Each file stores a struct
> containing the following fields for an image:
> 
> cjdata.label: 1 for meningioma, 2 for glioma, 3 for pituitary tumor
> cjdata.PID: patient ID
> cjdata.image: image data
> cjdata.tumorBorder: a vector storing the coordinates of discrete points on tumor border.
> 		For example, [x1, y1, x2, y2,...] in which x1, y1 are planar coordinates on tumor border.
> 		It was generated by manually delineating the tumor border. So we can use it to generate
> 		binary image of tumor mask.
> cjdata.tumorMask: a binary image with 1s indicating tumor region
> 
> -----
> This data was used in the following paper:
> 1. Cheng, Jun, et al. "Enhanced Performance of Brain Tumor Classification via Tumor Region Augmentation
> and Partition." PloS one 10.10 (2015).
> 2. Cheng, Jun, et al. "Retrieval of Brain Tumors by Adaptive Spatial Pooling and Fisher Vector 
> Representation." PloS one 11.6 (2016). Matlab source codes are available on github 
> https://github.com/chengjun583/brainTumorRetrieval
> 
> -----
> Jun Cheng
> School of Biomedical Engineering
> Southern Medical University, Guangzhou, China
> Email: chengjun583@qq.com

## Usage

For example, on Google Colab:

```
!wget https://github.com/BirkhoffLee/dataset-figshare/archive/refs/heads/main.zip
!unzip main.zip
!rm main.zip
!mv dataset-figshare-main brain_tumor_figshare
!tfds build brain_tumor_figshare
```

After it's built, you can use it as a normal TFDS dataset:

```python
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds

ds, info = tfds.load('brain_tumor_figshare', split='all', with_info=True)
tfds.as_dataframe(ds.take(4), info) # This takes 4 data and plots them
```

## License

The original dataset is licensed under the CC BY 4.0 License. If you use this dataset, please cite the original authors.
