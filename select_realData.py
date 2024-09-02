import pandas as pd

# # 读取txt文件，以'\t'分隔
# data = pd.read_csv("D:/PBMC/GEO/RNAseq_scRNAseq/E-MTAB/scRNA/pancreas_refseq_rpkms_counts_3514sc_2.txt", sep='\t')
#
# # 只保留前3516列的数据
# data = data.iloc[:, :3516]
# # 如果需要，可以将修改后的数据保存回文件
# data.to_csv("D:/PBMC/GEO/RNAseq_scRNAseq/E-MTAB/scRNA/your_new_file.txt", sep='\t', index=False)


file_path = 'D:/PBMC/GEO/RNAseq_scRNAseq/E-MTAB/scRNA/10000/'
# # ------------------删除全为0的行---------------------------------
# file_name = file_path + 'pancreas_refseq_rpkms_counts_3514sc_2.txt'
# # 读取TXT文件，以'\t'分隔
# data = pd.read_csv(file_name, sep='\t', index_col=0)
# # 删除数据全为'0'的行
# # data_filtered = data[(data != 0).any(axis=1)]
# # 删除行和小于等于0的行
# data_filtered = data[data.sum(axis=1) > 0]
#
# # 将结果保存到一个新的TXT文件
# # data_filtered.to_csv(file_path+"filtered_data.txt", sep='\t', index=False)
# data_filtered.to_csv(file_path+"filtered_data.txt", sep='\t', index=True, header=True)
# -----------去除重复的行索引-----------------------------------------------------------
# file_path = 'D:/PBMC/GEO/RNAseq_scRNAseq/E-MTAB/bulkRNA/'
# file_name = file_path + 'pancreas_refseq_rpkms_wholeislet_1.txt'
# # 读取TXT文件，以'\t'分隔，第一列作为行索引
# data = pd.read_csv(file_name, sep='\t', index_col=0)
# # 获取行索引
# row_names = data.index
# # 查找重复的行索引
# duplicates = row_names[row_names.duplicated(keep=False)]
#
# # 检查是否存在重复的行索引
# if data.index.duplicated().any():
#     print("Found duplicate row names. Keeping the first occurrence and removing others.")
#
#     # 保留第一次出现的行，删除其他重复行
#     data = data[~data.index.duplicated(keep='first')]
# else:
#     print("No duplicate row names found.")
#
# # 将处理后的数据保存回文件（可选）
# output_path = file_path + 'del_Duplicate.txt'  # 您可以自定义输出文件路径
# data.to_csv(output_path, sep='\t', index=True, header=True)

# -------------------------筛选出包含关键字的列-------------------------------------------------
file_name = file_path + 'scRNA_data_7_10000.txt'
# sample_key = 'HP1504101T2D'
# sample_key = ['HP1504101T2D', 'HP1504901', 'HP1506401', 'HP1507101', 'HP1508501T2D', 'HP1525301T2D', 'HP1526901T2D']
# sample_key = 'HP1526901T2D'
# sample_key = 'HP1525301T2D'
# sample_key = 'HP1508501T2D'
# sample_key = 'HP1504901'
# sample_key = 'HP1506401'
sample_key = 'HP1507101'
out_path = file_path + sample_key +'/' + sample_key +'_scData_10000.txt'

# 读取txt文件，以'\t'分隔
data = pd.read_csv(file_name, sep='\t', index_col=0)
# 筛选出列索引中包含'HP1504101T2D'的列
filtered_data = data.filter(like=sample_key)
# 打印筛选出的列数量
print(f"Number of columns containing {sample_key}: {filtered_data.shape[1]}\n")
# 将结果保存到一个新的txt文件
filtered_data.to_csv(out_path, sep='\t', index=True, header=True)



