from create_dataset import BulkCreate

"""
Simulation of artificial bulk RNA-seq samples from scRNA-seq data
and subsequenbt formatting in .h5ad file for training with Rescue
"""
import argparse


def create_dataset(simulate_dir, data_dir, sample_size, num_samples, pattern, fmt):
    unknown_celltypes=[]
    unknown_celltypes = list(unknown_celltypes)
    bulk_simulator = BulkCreate(sample_size=sample_size,
                                   num_samples=num_samples,
                                   data_path=data_dir,
                                   out_dir=simulate_dir,
                                   pattern=pattern,
                                   unknown_celltypes=unknown_celltypes,
                                   fmt=fmt)
    bulk_simulator.simulate()

if __name__ == "__main__":
    outPath = 'seurat_pbmc/13689'
    dataPath = 'seurat_pbmc/13689/data'
    # outPath = 'seurat_pbmc/4000/10000'
    # dataPath = 'seurat_pbmc/4000/10000/data'
    parser = argparse.ArgumentParser(
        description='Simulation of artificial bulk RNA-seq samples from scRNA-seq data for training or testing.')
    parser.add_argument('--out_path', type=str, default=outPath)
    parser.add_argument('--data_path', type=str, default=dataPath)
    parser.add_argument('--sample_size', type=int, default=500, help='Total number of cells')
    parser.add_argument('--sample_num', type=int, default=1000, help='Number of samples')
    parser.add_argument('--data_counts', type=str, default='*_counts.txt')
    parser.add_argument('--data_suffix', type=str, default='txt')
    args = parser.parse_args()

    create_dataset(
        simulate_dir=args.out_path,
        data_dir=args.data_path,
        sample_size=args.sample_size,
        num_samples =args.sample_num,
        pattern=args.data_counts,
        fmt=args.data_suffix,
    )




