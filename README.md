# Rescue: Resnet model employing scRNA-seq for characterizing cell composition by using expression of whole genome

<!--#![](https://github.com/xyMa00/Rescue/wiki/png/Rescue_model.png)-->
![](https://github.com/xyMa00/Rescue/wiki/png/Rescue_model.png)

## Installation  

Rescue neural network is implemented in [Pytorch](https://pytorch.org/) framework.  
Running Rescue on CUDA is recommended if available.   

#### Install from PyPI

    pip install rescue_dec

#### Install latest develop version from GitHub
    https://github.com/xyMa00/Rescue.git
or download and install

	git clone git@github.com:xyMa00/Rescue.git
	cd Rescue
	python setup.py install
    
Installation only requires a few minutes. 

 #### Simulation of bulk RNA-seq samples from scRNA-seq data 

    python create_dataset.py --out_path seurat_pbmc/4000/10000 --data_path seurat_pbmc/4000/10000/data --sample_num 4000
You need two .txt files about scRNA-seq data: *_celltypes.txt and _counts.txt.
(The _celltypes.txt file contains the 'Celltype' column.)

#### Correlation parameters 
* --out_path:the path of output.
* --data_path:the path of the scRNA-seq data.
* --sample_size:total number of cells.(default:500)
* --sample_num:number of samples.(default:4000)

## Quickly start
Using PBMC(3k) data as an example.\
(We provide detailed processing in **[Tutorial PBMC(3k)](https://github.com/xyMa00/Rescue/wiki/PBMC)** )
#### Input
* h5ad file(should contain 'cell_type' in its obs).

#### Run to train the model

    python Rescue.py --dataPath seurat_pbmc/4000/10000/train/pbmc3k_9_10000_4000.h5ad --testPath seurat_pbmc/4000/10000/test/pbmc3k_9_10000_1000.h5ad

#### Output
Output will be saved in the output folder including:
* **model.pt**:  saved model to predict cooperated with option --pretrain


#### Run to predict

    python Rescue.py --dataPath seurat_pbmc/4000/10000/train/pbmc3k_9_10000_4000.h5ad --testPath seurat_pbmc/4000/10000/test/pbmc3k_9_10000_1000.h5ad --modelPath pre/model_pbmc3k_9_10000_4000.pt --pretrain

#### Correlation parameters 
* --dataPath:the path of the training dataset.
* --testPath:the path of the testing dataset.
* --modelPath:the path of the pre-trained model.
* --outdir:the path of output.
* --pretrain:whether to load pretrained models. (default:False)
* --lr:learning rate.(default: 0.0001)
* --batch_size: batch size.(default: 32)
<!--
* save results in a specific folder: [--outdir] 
* modify the initial learning rate, default is 0.0001: [--lr]  
* you can change the batch size, default is 32: [--batch_size] 
-->

#### Help
Look for more usage of Rescue 

	Rescue.py --help 

Use functions in Rescue packages.

	import rescue
	from rescue import *

## Tutorial
**[Tutorial PBMC(3k)](https://github.com/xyMa00/Rescue/wiki/PBMC)**   :Run Rescue on scRNA-seq **PBMC(3k)** dataset (k=9, 2638 cells, 13714 genes)\
**[Tutorial Intestine](https://github.com/xyMa00/Rescue/wiki/Intestine)**   :Run Rescue on scRNA-seq **Intestine** dataset (k=8, 48055 cells, 2500 genes)\
**[Tutorial Mouse_kidney](https://github.com/xyMa00/Rescue/wiki/Mouse_kidney)**   :Run Rescue on scRNA-seq **Mouse_kidney** dataset (k=13, 9897 cells, 16129 genes)
