from simulation import BulkSimulator

"""
Simulation of artificial bulk RNA-seq samples from scRNA-seq data
and subsequenbt formatting in .h5ad file for training with Scaden
"""


def simulation(simulate_dir, data_dir, sample_size, num_samples, pattern,
               unknown_celltypes, out_prefix, fmt):

    unknown_celltypes = list(unknown_celltypes)
    bulk_simulator = BulkSimulator(sample_size=sample_size,
                                   num_samples=num_samples,
                                   data_path=data_dir,
                                   out_dir=simulate_dir,
                                   pattern=pattern,
                                   unknown_celltypes=unknown_celltypes,
                                   fmt=fmt)

    # Perform dataset simulation
    bulk_simulator.simulate()

    # Merge the resulting datasets
    # bulk_simulator.merge_datasets(data_dir=simulate_dir,
    #                               files=bulk_simulator.dataset_files,
    #                               out_name=out_prefix + ".h5ad")


if __name__ == "__main__":
    # simulation(simulate_dir='LUSC_IA3_BC3_10/2500', data_dir='LUSC_IA3_BC3_10/2500/data', sample_size=500,
    #            num_samples=100,
    #            pattern='*_counts.txt',
    #            unknown_celltypes=[], fmt="txt", out_prefix='exam')
    simulation(simulate_dir='seurat_pbmc/4000/10000', data_dir='seurat_pbmc/4000/10000/data', sample_size=500,
               num_samples=4000,
               pattern='*_counts.txt',
               unknown_celltypes=[], fmt="txt", out_prefix='exam')



