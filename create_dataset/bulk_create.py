import logging
import glob
import os
import sys
import gc

import pandas as pd
import anndata as ad
import numpy as np

from rich.progress import BarColumn, Progress

logger = logging.getLogger(__name__)


# 自定义排序键函数
def sort_key(s):
    import re
    # 分离字符串中的字母部分和数字部分
    match = re.match(r"([a-zA-Z]+)([0-9]+)", s)
    if match:
        letter_part = match.group(1)
        number_part = int(match.group(2))
        return (letter_part, number_part)
    return (s, 0)

def generate_numbers_with_fixed_sum(fixed_number, n, unk_num=-1):
    if fixed_number < 0 or fixed_number > 1:
        raise ValueError("The fixed number must be between 0 and 1.")
    if n < 1:
        raise ValueError("The total number of elements n must be at least 1.")
    # Generate n-1 random numbers
    random_numbers = np.random.rand(n - 1)
    # Calculate the sum of the random numbers
    random_sum = np.sum(random_numbers)
    # Normalize the random numbers to sum to (1 - fixed_number)
    random_numbers = (random_numbers / random_sum) * (1 - fixed_number)
    # Combine the fixed number with the normalized random numbers
    if unk_num>=0:
        # 在第 unk_num 个位置插入数字，索引为 unk_num-1（0 基）
        result = np.insert(random_numbers, unk_num-1, fixed_number)
    else:
        result = np.append(fixed_number, random_numbers)

    return result




def create_fractions_unk(no_celltypes, count, n_samps=4000):
    """
    Create random fractions
    :param no_celltypes: number of fractions to create
    :return: list of random fractions of length no_celltypes
    """
    # fracs = np.random.rand(no_celltypes)
    # fracs_sum = np.sum(fracs)
    # fracs = np.divide(fracs, fracs_sum)

    if no_celltypes == 1:
        fixed_number = 1
    else:
        for i in range(1, 11):
            if (i - 1) * 0.1 * n_samps < count <= i * 0.1 * n_samps:
                fixed_number = np.random.uniform((i - 1) * 0.1, i * 0.1)
                break
        # fixed_number = np.random.uniform(0.5, 1)
        print(fixed_number)
        # Example usage
        # fixed_number = lie1

    # # # # # # 指定占比
    # fixed_number = 0.3
    # # fixed_number = np.random.uniform(0.5, 1)
    # print(fixed_number)
    # Example usage
    # fixed_number = lie1
    n = no_celltypes  # Total number of elements
    fracs = generate_numbers_with_fixed_sum(fixed_number, n, 6)
    print("Numbers:", fracs)
    print("Sum of numbers:", np.sum(fracs))

    return fracs
def create_fractions(no_celltypes, count, n_samps=2000):
    """
    Create random fractions
    :param no_celltypes: number of fractions to create
    :return: list of random fractions of length no_celltypes
    """
    # # # # # 指定占比
    # fixed_number = np.random.uniform(0, 0.1)
    # # fixed_number = np.random.uniform(0.5, 1)
    # print(fixed_number)
    # # Example usage
    # # fixed_number = lie1
    # n = no_celltypes  # Total number of elements
    # fracs = generate_numbers_with_fixed_sum(fixed_number, n)
    # print("Numbers:", fracs)
    # print("Sum of numbers:", np.sum(fracs))

    # # # # 指定占比
    # if count <= 0.1*n_samps:
    #     fixed_number = np.random.uniform(0, 0.1)
    # if 0.1*n_samps< count <= 0.2 * n_samps:
    #     fixed_number = np.random.uniform(0.1, 0.2)
    # fixed_number=0

    if no_celltypes==1:
        fixed_number = 1
    else:
        for i in range(1, 11):
            if (i - 1) * 0.1 * n_samps < count <= i * 0.1 * n_samps:
                fixed_number = np.random.uniform((i - 1) * 0.1, i * 0.1)
                break
        # fixed_number = np.random.uniform(0.5, 1)
        print(fixed_number)
        # Example usage
        # fixed_number = lie1
    n = no_celltypes  # Total number of elements
    fracs = generate_numbers_with_fixed_sum(fixed_number, n)
    print("Numbers:", fracs)
    print("Sum of numbers:", np.sum(fracs))
    return fracs

def create_fractions_2(no_celltypes, count, n_samps=2000):
    """
    Create random fractions
    :param no_celltypes: number of fractions to create
    :return: list of random fractions of length no_celltypes
    """
    for i in range(1, 11):
        if (i - 1) * 0.1 * n_samps < count <= i * 0.1 * n_samps:
            fixed_number = np.random.uniform((i - 1) * 0.1, i * 0.1)
            break
    # fixed_number = np.random.uniform(0.5, 1)
    print(fixed_number)
    # Example usage
    # fixed_number = lie1
    n = no_celltypes  # Total number of elements
    pos = count%no_celltypes+1
    print('pos:', pos)
    fracs = generate_numbers_with_fixed_sum(fixed_number, n, pos)
    print("Numbers:", fracs)
    print("Sum of numbers:", np.sum(fracs))
    return fracs


class BulkCreate(object):
    """
    BulkSimulator class for the simulation of artificial bulk samples
    from scRNA-seq datasets

    :param sample_size: number of cells per sample
    :param num_samples: number of sample to simulate
    :param data_path: path to the data directory
    :param out_dir: output directory
    :param pattern of the data files
    :param unknown_celltypes: which celltypes to merge into the unknown class
    :param fmt: the format of the input files, can be txt or h5ad
    """

    def __init__(
        self,
        sample_size=100,
        num_samples=1000,
        data_path="./",
        out_dir="./",
        pattern="*_counts.txt",
        unknown_celltypes=None,
        fmt="txt",
    ):
        if unknown_celltypes is None:
            unknown_celltypes = ["unknown"]

        self.sample_size = sample_size
        self.num_samples = num_samples // 2
        # self.num_samples = num_samples
        self.data_path = data_path
        self.out_dir = out_dir
        self.pattern = pattern
        self.unknown_celltypes = unknown_celltypes
        self.format = fmt
        self.datasets = []
        self.dataset_files = []

    def simulate(self):
        """simulate artificial bulk datasets"""
        # List available datasets
        if not self.data_path.endswith("/"):
            self.data_path += "/"
        files = glob.glob(os.path.join(self.data_path, self.pattern))
        files = [os.path.basename(x) for x in files]
        self.datasets = [x.replace(self.pattern.replace("*", ""), "") for x in files]
        self.dataset_files = [
            os.path.join(self.out_dir, x + ".h5ad") for x in self.datasets
        ]

        if len(self.datasets) == 0:
            logging.error(
                "No datasets found! Have you specified the pattern correctly?"
            )
            sys.exit(1)

        logger.info("Datasets: [cyan]" + str(self.datasets) + "[/]")

        # Loop over datasets and simulate bulk data
        for i, dataset in enumerate(self.datasets):
            gc.collect()
            logger.info(f"[bold u]Simulating data from {dataset}")
            self.simulate_dataset(dataset)

        logger.info("[bold green]Finished data simulation!")

    def simulate_dataset(self, dataset):
        """
        Simulate bulk data from a single scRNA-seq dataset
        @param dataset:
        @type dataset:
        @return:
        @rtype:
        """

        # load the dataset
        data_x, data_y = self.load_dataset(dataset)

        # Merge unknown celltypes
        logger.info(f"Merging unknown cell types: {self.unknown_celltypes}")
        data_y = self.merge_unknown_celltypes(data_y)

        logger.info(f"Subsampling [bold cyan]{dataset}[/] ...")

        # Extract celltypes
        celltypes = list(set(data_y["Celltype"].tolist()))
        available_celltypes = sorted(celltypes, key=sort_key)
        print('available_celltypes:', available_celltypes)
        celltypes = available_celltypes
        tmp_x, tmp_y = self.create_subsample_dataset(
            data_x, data_y, celltypes=celltypes
        )

        tmp_x = tmp_x.sort_index(axis=1)
        ratios = pd.DataFrame(tmp_y, columns=celltypes)
        ratios["ds"] = pd.Series(np.repeat(dataset, tmp_y.shape[0]), index=ratios.index)

        ann_data = ad.AnnData(
            X=tmp_x.to_numpy(),
            obs=ratios,
            var=pd.DataFrame(columns=[], index=list(tmp_x)),
        )
        ann_data.uns["unknown"] = self.unknown_celltypes
        ann_data.uns["cell_types"] = celltypes

        # ann_data.write(os.path.join(self.out_dir, dataset + ".h5ad"))
        ann_data.write(os.path.join(self.out_dir, dataset + ".h5ad"), compression='gzip')

    def load_dataset(self, dataset):
        """
        Load a dataset
        @param dataset:
        @type dataset:
        @return:
        @rtype:
        """
        pattern = self.pattern.replace("*", "")
        logger.info(f"Loading [cyan]{dataset}[/] dataset ...")
        dataset_counts = dataset + pattern
        dataset_celltypes = dataset + "_celltypes.txt"

        # Load data in .txt format
        if self.format == "txt":
            # Try to load celltypes
            try:
                y = pd.read_table(os.path.join(self.data_path, dataset_celltypes))
                # Check if has Celltype column
                if "Celltype" not in y.columns:
                    logger.error(
                        f"No 'Celltype' column found in {dataset}_celltypes.txt! Please make sure to include this "
                        f"column. "
                    )
                    sys.exit()
            except FileNotFoundError as e:
                logger.error(
                    f"No celltypes file found for [cyan]{dataset}[/]. It should be called [cyan]{dataset}_celltypes.txt."
                )
                sys.exit(e)

            # Try to load data file
            try:
                x = pd.read_table(
                    os.path.join(self.data_path, dataset_counts),
                    index_col=0,
                    dtype=np.float32,
                )
            except FileNotFoundError as e:
                logger.error(
                    f"No counts file found for [cyan]{dataset}[/]. Was looking for file [cyan]{dataset_counts}[/]"
                )
                sys.exit(e)

            # Check that celltypes and count file have the same number of cells
            if not y.shape[0] == x.shape[0]:
                logger.error(
                    f"Different number of cells in {dataset}_celltypes and {dataset_counts}! Make sure the data has "
                    f"been processed correctly. "
                )
                sys.exit(1)

        # Load data in .h5ad format
        elif self.format == "h5ad":
            try:
                data_h5ad = ad.read_h5ad(os.path.join(self.data_path, dataset_counts))
            except FileNotFoundError as e:
                logger.error(
                    f"No h5ad file found for [cyan]{dataset}[/]. Was looking for file [cyan]{dataset_counts}"
                )
                sys.exit(e)
            # cell types
            try:
                y = pd.DataFrame(data_h5ad.obs.Celltype)
                y.reset_index(inplace=True, drop=True)
            except Exception as e:
                logger.error(f"Celltype attribute not found for [cyan]{dataset}")
                sys.exit(e)
            # counts
            x = pd.DataFrame(data_h5ad.X.todense())
            x.index = data_h5ad.obs_names
            x.columns = data_h5ad.var_names
            del data_h5ad
        else:
            logger.error(f"Unsupported file format {self.format}!")
            sys.exit(1)

        return x, y

    def merge_unknown_celltypes(self, y):
        """
        Merge all unknown celltypes together
        :param y: list of cell type labels
        :return: list of cell types with unknown cell types merged into "Unknown" type
        """
        celltypes = list(y["Celltype"])
        new_celltypes = [
            "Unknown" if x in self.unknown_celltypes else x for x in celltypes
        ]
        y["Celltype"] = new_celltypes
        return y

    def create_subsample_dataset(self, x, y, celltypes):
        """
        Generate many artifial bulk samples with known fractions
        This function will create normal and sparse samples (no_samples)
        @param x:
        @param y:
        @param celltypes:
        @return:
        """
        sim_x = []
        sim_y = []

        # Create normal samples
        progress_bar = Progress(
            "[bold blue]{task.description}",
            "[bold cyan]{task.fields[samples]}",
            BarColumn(bar_width=None),
        )
        with progress_bar:
            normal_samples_progress = progress_bar.add_task(
                "Normal samples", total=self.num_samples, samples=0
            )
            sparse_samples_progress = progress_bar.add_task(
                "Sparse samples", total=self.num_samples, samples=0
            )
            n_sam = self.num_samples
            for i in range(self.num_samples):
                progress_bar.update(normal_samples_progress, advance=1, samples=i + 1)
                # print(i)
                sample, label = self.create_subsample(x, y, celltypes, sparse=False, samp = n_sam, count=i+1)
                sim_x.append(sample)
                sim_y.append(label)
            # Create sparase samples
            for i in range(self.num_samples):
                progress_bar.update(sparse_samples_progress, advance=1, samples=i + 1)
                # sample, label = self.create_subsample(x, y, celltypes, sparse=True)
                sample, label = self.create_subsample(x, y, celltypes, sparse=True, samp=n_sam, count=i + 1)
                sim_x.append(sample)
                sim_y.append(label)

        sim_x = pd.concat(sim_x, axis=1).T
        sim_y = pd.DataFrame(sim_y, columns=celltypes)

        return sim_x, sim_y

    def create_subsample(self, x, y, celltypes, sparse=False, samp = 2000, count=0):
        """
        Generate artifical bulk subsample with random fractions of celltypes
        If sparse is set to true, add random celltypes to the missing celltypes

        @param x:
        @param y:
        @param celltypes:
        @param sparse:
        @return:
        """
        available_celltypes = celltypes
        if sparse:
            no_keep = np.random.randint(1, len(available_celltypes))
            # no_keep = np.random.randint(1, len(available_celltypes)+1)
            keep = np.random.choice(
                list(range(len(available_celltypes))), size=no_keep, replace=False
            )
            available_celltypes = [available_celltypes[i] for i in keep]

            no_avail_cts = len(available_celltypes)
            fracs = create_fractions(no_celltypes=no_avail_cts, count=count, n_samps=samp)
        else:
            # print(available_celltypes)
            no_avail_cts = len(available_celltypes)
            # Create fractions for available celltypes
            fracs = create_fractions_2(no_celltypes=no_avail_cts, count=count, n_samps=samp)
        samp_fracs = np.multiply(fracs, self.sample_size)
        samp_fracs = list(map(int, samp_fracs))

        # Make complete fracions
        fracs_complete = [0] * len(celltypes)
        for i, act in enumerate(available_celltypes):
            idx = celltypes.index(act)
            fracs_complete[idx] = fracs[i]

        artificial_samples = []
        for i in range(no_avail_cts):
            ct = available_celltypes[i]
            # 从数据框 x 中筛选出细胞类型为 ct 的所有行
            cells_sub = x.loc[np.array(y["Celltype"] == ct), :]
            # 随机选择一些行的索引，数量由 samp_fracs[i] 决定，这个数量代表需要抽取的样本比例。
            cells_fraction = np.random.randint(0, cells_sub.shape[0], samp_fracs[i])
            cells_sub = cells_sub.iloc[cells_fraction, :]
            artificial_samples.append(cells_sub)
        df_samp = pd.concat(artificial_samples, axis=0)
        df_samp = df_samp.sum(axis=0)
        return df_samp, fracs_complete


    @staticmethod
    def merge_datasets(data_dir="./", files=None, out_name="data.h5ad"):
        """

        @param out_name: name of the merged .h5ad file
        @param data_dir: directory to look for datasets
        @param files: list of files to merge
        @return:
        """
        non_celltype_obs = ["ds", "batch"]
        if not files:
            files = glob.glob(os.path.join(data_dir, "*.h5ad"))

        logger.info(f"Merging datasets: {files} into [bold cyan]{out_name}")

        # load first file
        adata = ad.read_h5ad(files[0])

        for i in range(1, len(files)):
            adata = adata.concatenate(ad.read_h5ad(files[i]), uns_merge="same")

        combined_celltypes = list(adata.obs.columns)
        combined_celltypes = [
            x for x in combined_celltypes if not x in non_celltype_obs
        ]
        for ct in combined_celltypes:
            adata.obs[ct].fillna(0, inplace=True)

        adata.uns["cell_types"] = combined_celltypes
        print(adata.obs.columns)
        adata.write(out_name, compression='gzip')
