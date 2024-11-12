# Rescue: Resnet model employing scRNA-seq for characterizing cell composition by using expression of whole genome

#![](https://github.com/xyMa00/Rescue/wiki/png/Rescue_model.png)


## Installation  

Rescue neural network is implemented in [Pytorch](https://pytorch.org/) framework.  
Running Rescue on CUDA is recommended if available.   

#### Install from PyPI

    pip install Rescue

#### Install latest develop version from GitHub
    https://github.com/xyMa00/Rescue.git
or download and install

	git clone git@github.com:xyMa00/Rescue.git
	cd Ressac
	python setup.py install
    
Installation only requires a few minutes. 

 #### From scRNA-seq generate simulation data sets by running 

    python create_dataset.py --out_path seurat_pbmc/4000/10000 --data_path seurat_pbmc/4000/10000/data
But you need two .txt files about scRNA-seq data: *_celltypes.txt and _counts.txt.
(The _celltypes.txt file contains the 'Celltype' column.)



## Data preprocessing
* First you need to convert the input data to a .h5ad file(this file should contain 'cell_type' in its obs).
* Second, you can screen based on cells and peaks, at the same time, you can choose to keep the most variable features.
* Third, the final number of retained peaks should be the square of some number.

You can refer to **epi_h5ad.py** for the whole process.


## Quickly start to make predictions

#### Input
* h5ad file(should contain 'cell_type' in its obs).

#### Run to train the model

    python Rescue.py --dataPath seurat_pbmc/4000/10000/train/pbmc3k_9_10000_4000.h5ad --testPath seurat_pbmc/4000/10000/test/pbmc3k_9_10000_1000.h5ad

#### Output
Output will be saved in the output folder including:
* **model.pt**:  saved model to predict cooperated with option --pretrain


#### Run to predict

    python Rescue.py --dataPath seurat_pbmc/4000/10000/train/pbmc3k_9_10000_4000.h5ad --testPath seurat_pbmc/4000/10000/test/pbmc3k_9_10000_1000.h5ad --modelPath pre/model_pbmc3k_9_10000_4000.pt --pretrain



#### Useful options  
* save results in a specific folder: [--outdir] 
* modify the initial learning rate, default is 0.0001: [--lr]  
* you can change the batch size, default is 32: [--batch_size] 


#### Help
Look for more usage of Rescue 

	Rescue.py --help 

Use functions in Rescue packages.

	import rescue
	from rescue import *

## Tutorial
**[Tutorial PBMC](https://github.com/xyMa00/Rescue/wiki/PBMC)**   :Run Rescue on scRNA-seq **PBMC** dataset (k=9, 13714 genes)


**[Tutorial LUSC](https://github.com/xyMa00/Rescue/wiki/LUSC)**   :Run Rescue on scRNA-seq **LUSC** dataset (k=10, 18081 genes)
