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
    bulk_simulator.merge_datasets(data_dir=simulate_dir,
                                  files=bulk_simulator.dataset_files,
                                  out_name=out_prefix + ".h5ad")


if __name__ == "__main__":
    # simulation(simulate_dir='LUSC_IA3_BC3_10/property/4000', data_dir='LUSC_IA3_BC3_10/data', sample_size=500,
    #            num_samples=4000,
    #            pattern='*_counts.txt',
    #            unknown_celltypes=[], fmt="txt", out_prefix='exam')
    # simulation(simulate_dir='E-MTAB/10000/HP1507101/4000', data_dir='E-MTAB/10000/HP1507101/data', sample_size=500,
    #            num_samples=5000,
    #            pattern='*_counts.txt',
    #            unknown_celltypes=[], fmt="txt", out_prefix='exam')
    # simulation(simulate_dir='LUSC_IA3_BC3_10/2500', data_dir='LUSC_IA3_BC3_10/2500/data',
    #            sample_size=500,
    #            num_samples=2000,
    #            pattern='*_counts.txt',
    #            unknown_celltypes=[], fmt="txt", out_prefix='exam')
    # simulation(simulate_dir='seurat_pbmc/20000/train_dataset', data_dir='seurat_pbmc/data',
    #            sample_size=500,
    #            num_samples=8000,
    #            pattern='*_counts.txt',
    #            unknown_celltypes=[], fmt="txt", out_prefix='exam')
    # simulation(simulate_dir='LUSC_10_types/4000', data_dir='LUSC_10_types/data', sample_size=500,
    #            num_samples=4000,
    #            pattern='*_counts.txt',
    #            unknown_celltypes=[], fmt="txt", out_prefix='exam')
    # simulation(simulate_dir='hscr/all_2_seurat_object_common_8/Z_S_HSCR3_dilated_2500/20000/4000', data_dir='hscr/all_2_seurat_object_common_8/Z_S_HSCR3_dilated_2500/data', sample_size=500,
    #            num_samples=4000,
    #            pattern='*_counts.txt',
    #            unknown_celltypes=[], fmt="txt", out_prefix='exam')
    simulation(simulate_dir='GSE257541/10000/pbmc48h/10000',
               data_dir='GSE257541/10000/pbmc48h/data', sample_size=500,
               num_samples=10000,
               pattern='*_counts.txt',
               unknown_celltypes=[], fmt="txt", out_prefix='exam')



