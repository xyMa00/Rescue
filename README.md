# Rescue: Resnet model employing scRNA-seq for characterizing cell composition by using expression of whole genome

#![](https://github.com/xyMa00/Rescue/wiki/png/Rescue_model.png)


## Installation  

Ressac neural network is implemented in [Pytorch](https://pytorch.org/) framework.  
Running Rescue on CUDA is recommended if available.   

#### install from PyPI

    pip install Rescue

#### install latest develop version from GitHub
    https://github.com/xyMa00/Rescue.git
or download and install

	git clone git@github.com:xyMa00/Rescue.git
	cd Ressac
	python setup.py install
    
Installation only requires a few minutes. 

 #### From scRNA-seq generate simulation data sets by running 

    simulate.py -d [input]
But you need two .txt files about scRNA-seq data: *_celltypes.txt and _counts.txt.
(The _celltypes.txt file contains the 'Celltype' column.)



## Data preprocessing
* First you need to convert the input data to a .h5ad file(this file should contain 'cell_type' in its obs).
* Second, you can screen based on cells and peaks, at the same time, you can choose to keep the most variable features.
* Third, the final number of retained peaks should be the square of some number.

You can refer to **epi_h5ad.py** for the whole process.


## Quick Start

#### Input
* h5ad file(should contain 'cell_type' in its obs).

#### Run 

    Rescue.py -d [input]

#### Output
Output will be saved in the output folder including:
* **model.pt**:  saved model to reproduce results cooperated with option --pretrain


#### Useful options  
* save results in a specific folder: [-o] or [--outdir] 
* modify the initial learning rate, default is 0.0001: [--lr]  
* change random seed for parameter initialization, default is 32: [--seed]


#### Help
Look for more usage of Rescue 

	Rescue.py --help 

Use functions in Rescue packages.

	import rescue
	from rescue import *

## Tutorial
**[Tutorial LUSC](https://github.com/xyMa00/Rescue/wiki/LUSC)**   Run Rescue on scRNA-seq **LUSC** dataset (k=10, 2500 genes)
