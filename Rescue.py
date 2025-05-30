"""
# File Name: Rescue.py
# Description: An attention-driven deep learning framework for cell type deconvolution in bulk RNA-seq.
    Input: 
        bulk RNA-seq data.
    Output:
        Proportion of cell types.
"""

import argparse
from rescue_dec import some_function



dataPath = 'seurat_pbmc/10000/train/pbmc3k_9_10000_4000.h5ad'
testPath = 'seurat_pbmc/10000/test/pbmc3k_9_10000_1000.h5ad'
modelPath = 'pre/model_pbmc3k_9_10000_4000.pt'
# dataPath = 'mouse_kidney/4000/train/mouse_kidney_9897_10000_4000.h5ad'
# testPath = 'mouse_kidney/4000/test/mouse_kidney_9897_10000_1000.h5ad'
# modelPath = ''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rescue: Resnet model employing scRNA-seq for characterizing cell composition by using expression of whole genome')
    parser.add_argument('--dataPath', type=str, default=dataPath)
    parser.add_argument('--testPath', type=str, default=testPath)
    parser.add_argument('--modelPath', type=str, default=modelPath)
    parser.add_argument('--outdir', type=str, default='output/', help='Output path')
    parser.add_argument('--pretrain', action='store_true', help='Load the trained model(default: False)')
    # parser.add_argument('--pretrain', action='store_false', help='Load the trained model(default: True)')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--gpu', default=0, type=int, help='Select gpu device number when training')
    parser.add_argument('--seed', type=int, default=18, help='Random seed for repeat results')
    args = parser.parse_args()

    adata = some_function(
        args.dataPath,
        args.testPath,
        args.modelPath,
        outdir=args.outdir,
        pretrain=args.pretrain,
        lr=args.lr,
        batch_size=args.batch_size,
        gpu=args.gpu,
        seed=args.seed,
    )
