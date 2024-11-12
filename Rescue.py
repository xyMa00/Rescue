"""
# File Name: Rescue.py
# Description: Resnet model employing scRNA-seq for characterizing cell composition by using expression of whole genome.
    Input: 
        bulk RNA-seq data.
    Output:
        Proportion of cell types.
"""

import argparse
from rescue import some_function


# dataPath = 'LUSC_IA3_BC3_10/2500/train/LUSC_IA3_BC3_10_2500_5000_4000.h5ad'
# testPath = 'LUSC_IA3_BC3_10/2500/test/LUSC_IA3_BC3_10_2500_2000.h5ad'
dataPath = 'seurat_pbmc/4000/10000/train/pbmc3k_9_10000_4000.h5ad'
testPath = 'seurat_pbmc/4000/10000/test/pbmc3k_9_10000_1000.h5ad'
modelPath = 'pre/model_pbmc3k_9_10000_4000.pt'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rescue: Resnet model employing scRNA-seq for characterizing cell composition by using expression of whole genome')
    parser.add_argument('--data_list', '-data', type=str, default=dataPath)
    parser.add_argument('--test_list', '-test', type=str, default=testPath)
    parser.add_argument('--model_path', '-model', type=str, default=modelPath)
    # parser.add_argument('--n_centroids', '-k', type=int, help='cluster number', default=9)
    parser.add_argument('--outdir', '-o', type=str, default='output/', help='Output path')
    parser.add_argument('--pretrain', action='store_true', help='Load the trained model(default: False)')
    # parser.add_argument('--pretrain', action='store_false', help='Do not load the trained model')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='Batch size')
    parser.add_argument('--gpu', '-g', default=0, type=int, help='Select gpu device number when training')
    parser.add_argument('--seed', type=int, default=18, help='Random seed for repeat results')
    args = parser.parse_args()

    adata = some_function(
        args.data_list,
        args.test_list,
        args.model_path,
        outdir=args.outdir,
        pretrain=args.pretrain,
        lr=args.lr,
        batch_size=args.batch_size,
        gpu=args.gpu,
        seed=args.seed,
    )
