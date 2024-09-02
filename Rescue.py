"""
# File Name: Rescue.py
# Description: Resnet based single-cell ATAC-seq clustering.
    Input: 
        scATAC-seq data
    Output:
        1. latent feature
        2. cluster assignment
        3. imputation data
"""

import argparse
from rescue import some_function

# dataPath='adata_train.h5ad'

#dataPath = 'adata_train_25281.h5ad'
# dataPath = 'adata_pre_25281.h5ad'
# dataPath = 'adata_pre.h5ad'
# dataPath = 'find_genes_adata_101124.h5ad'
#dataPath = 'adata_mouse_train_16129.h5ad'

#dataPath = 'LUSC_IA3_BC8/adata_LUSC_IA3_BC8_17956.h5ad'
#dataPath = 'LUSC/adata_LUSC_IA3_BC3_17956.h5ad'
#dataPath = 'seurat_pbmc_5000/adata_pbmc_13689.h5ad'
#dataPath = 'marge_10/combined_train.h5ad'

# dataPath = 'hscr_marge_8/combined_train.h5ad'
# testPath = 'hscr_marge_8/combined_test.h5ad'
# dataPath = 'hscr/all_2_seurat_object_common_8/Z_S_HSCR2_dilated_2500/need/adata_Z_S_HSCR2_dilated_2500_pre.h5ad'
# dataPath = 'hscr/all_2_seurat_object_common_8/Z_S_HSCR3_dilated_2500/need/adata_Z_S_HSCR3_dilated_2500_pre.h5ad'
# dataPath = 'hscr/all_2_seurat_object_common_8/commom_8_2/Z_S_HSCR2_dilated_2500_2/need/adata_Z_S_HSCR2_dilated_2500_2.h5ad'
# dataPath = 'hscr/all_2_seurat_object_common_8/commom_8_2/Z_S_HSCR3_dilated_2500_2/need/adata_Z_S_HSCR3_dilated_2500_2.h5ad'
# dataPath = 'LUSC_IA3_BC3_10/20000/need/train_dataset/adata_LUSC_IA3_BC3_10_train_17956_400.h5ad'
# testPath = 'LUSC_IA3_BC3_10/20000/need/adata_LUSC_IA3_BC3_10_pre_17956.h5ad'
# testPath = 'LUSC_10_types/marge_20000/combined_test.h5ad'
# testPath = 'hscr/all_2_seurat_object_common_8/20000/combined_test.h5ad'
# testPath = 'hscr/all_2_seurat_object_common_8/Z_S_HSCR3_dilated_2500/20000/test_2500_20000_4000.h5ad'

# testPath = 'LUSC_IA3_BC3_10/rare/200/LUSC_IA3_BC3_10_test_percent50_100.h5ad'
# dataPath = 'LUSC_IA3_BC3_10/rare/200/LUSC_IA3_BC3_10_train_percent10_400.h5ad'

# testPath = 'LUSC_IA3_BC3_10/rare/200/LUSC_IA3_BC3_10_test_percent50_50_17956.h5ad'
dataPath = 'LUSC_IA3_BC3_10/rare/200/LUSC_IA3_BC3_10_train_percent10_200_17956.h5ad'
# testPath = 'LUSC_IA3_BC3_10/rare/LUSC_IA3_BC3_10_test_percent50_1000_17956.h5ad'
# testPath = 'LUSC_IA3_BC3_10/property/LUSC_IA3_BC3_10_2000_17956.h5ad'
# testPath = 'LUSC_IA3_BC8_10/property/LUSC_IA3_BC8_10_2000_17956.h5ad'
# testPath = 'HP1504101T2D/real/RNA_data_new.h5ad'
# testPath = 'hscr/all_2_seurat_object_common_8/Z_S_HSCR3_dilated_2500/need/adata_Z_S_HSCR3_dilated_2500_pre.h5ad'
# testPath = 'seurat_pbmc/4000/pbmc_1000_13689.h5ad'
# testPath = 'LUSC_IA3_BC8/need/adata_LUSC_IA3_BC8_pre_17956.h5ad'
# testPath = 'mouse/4000/adata_mouse_test_16129_5000_1000.h5ad'
# testPath = 'LUSC_IA3_BC8/2500/4000/LUSC_IA3_BC8_common_2500_5000_1000.h5ad'
# testPath = 'LUSC_IA3_BC3_10/2500/4000/LUSC_IA3_BC3_10_2500_5000_1000.h5ad'
# testPath = 'LUSC_IA3_BC3_10/2500/LUSC_IA3_BC3_10_2500_2000.h5ad'
# testPath = 'E-MTAB/10000/HP1507101/RNA_data_new.h5ad'
# testPath = 'seurat_pbmc/20000/pbmc_4000_13689.h5ad'
testPath = 'GSE257541/10000/pbmc0h_1/RNA_data_new_0h_1.h5ad'




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rescue: Resnet based single-cell ATAC-seq clustering')
    parser.add_argument('--data_list', '-d', type=str, nargs='+', default=dataPath)
    parser.add_argument('--test_list', '-t', type=str, nargs='+', default=testPath)
    # parser.add_argument('--n_centroids', '-k', type=int, help='cluster number', default=40)
    parser.add_argument('--n_centroids', '-k', type=int, help='cluster number', default=6)
    # parser.add_argument('--n_centroids', '-k', type=int, help='cluster number', default=13)
    parser.add_argument('--outdir', '-o', type=str, default='output/', help='Output path')
    parser.add_argument('--verbose', action='store_true', help='Print loss of training process')
    # parser.add_argument('--pretrain', type=str, default=None, help='Load the trained model')
    parser.add_argument('--pretrain', type=str, default=True, help='Load the trained model')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate')
    # parser.add_argument('--lr', type=float, default=0.003, help='Learning rate')
    #parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='Batch size')
    # parser.add_argument('--batch_size', '-b', type=int, default=1, help='Batch size')
    parser.add_argument('--gpu', '-g', default=0, type=int, help='Select gpu device number when training')
    parser.add_argument('--seed', type=int, default=18, help='Random seed for repeat results')
    parser.add_argument('--encode_dim', type=int, nargs='*', default=[1024, 128], help='encoder structure')
    parser.add_argument('--decode_dim', type=int, nargs='*', default=[], help='encoder structure')
    parser.add_argument('--latent', '-l',type=int, default=8, help='latent layer dim')
    parser.add_argument('--min_peaks', type=float, default=1000, help='Remove low quality cells with few peaks')
    parser.add_argument('--min_cells', type=float, default=5, help='Remove low quality peaks')
    parser.add_argument('--n_feature', type=int, default=100000, help='Keep the number of highly variable peaks')
    parser.add_argument('--log_transform', action='store_true', help='Perform log2(x+1) transform')
    parser.add_argument('--max_iter', '-i', type=int, default=30000, help='Max iteration')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--impute', action='store_true', help='Save the imputed data in layer impute')
    parser.add_argument('--binary', action='store_true', help='Save binary imputed data in layer binary')
    parser.add_argument('--embed', type=str, default='UMAP')
    #parser.add_argument('--embed', type=str, default='tSNE')
    parser.add_argument('--reference', type=str, default='celltype')
    # parser.add_argument('--cluster_method', type=str, default='leiden')
    parser.add_argument('--cluster_method', type=str, default='kmeans')

    args = parser.parse_args()

    adata = some_function(
        args.data_list,
        args.test_list,
        n_centroids = args.n_centroids,
        outdir = args.outdir,
        verbose = args.verbose,
        pretrain = args.pretrain,
        lr = args.lr,
        batch_size = args.batch_size,
        gpu = args.gpu,
        seed = args.seed,
        encode_dim = args.encode_dim,
        decode_dim = args.decode_dim,
        # latent = args.latent,
        latent=args.n_centroids,
        min_peaks = args.min_peaks,
        min_cells = args.min_cells,
        n_feature = args.n_feature,
        log_transform = args.log_transform,
        max_iter = args.max_iter,
        weight_decay = args.weight_decay,
        impute = args.impute,
        binary = args.binary,
        embed = args.embed,
        # reference = args.reference,
        # cluster_method = args.cluster_method,
    )

    