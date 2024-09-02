import pandas as pd
import numpy as np

# 读取TXT文件
# file_path = 'LUSC_IA3_BC3_10/property/200/obs_data_train_counts.txt'
# file_path = 'LUSC_IA3_BC3_10/property/200/test1000/obs_data_counts.txt'

# file_path = 'LUSC_IA3_BC3_10/20000/need/train_dataset/obs_data_counts.txt'
file_path = 'LUSC_IA3_BC3_10/2500/4000//pred/truth_data.txt'

data = pd.read_csv(file_path, sep='\t', index_col=0)
# # 第一行已经作为列名，移除第一行
# data.columns = data.iloc[0]
# data = data[1:]
# 将数据类型转换为数值类型，以便统计数值分布
data = data.apply(pd.to_numeric)
# 定义区间
bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
labels = [f'{bins[i]}-{bins[i+1]}' if i == 0 else f'({bins[i]}-{bins[i+1]}]' for i in range(len(bins)-1)]

# 统计每列数据的分布情况
distribution = pd.DataFrame()
for col in data.columns:
    distribution[col] = pd.cut(data[col], bins=bins, labels=labels, include_lowest=True).value_counts().reindex(labels, fill_value=0)
# 打印分布情况
print(distribution)


# 根据分布绘制箱型图
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取TXT文件
data_path = 'LUSC_IA3_BC3_10/2500/4000/pred/truth_data'
file_path = data_path + '.txt'
data = pd.read_csv(file_path, sep='\t', index_col=0)

# 将数据类型转换为数值类型
data = data.apply(pd.to_numeric)

# 统计每列数据在 [0, 1] 范围内的分布
data_in_range = data[(data >= 0) & (data <= 1)]

# 绘制箱型图
plt.figure(figsize=(10, 6))
sns.boxplot(data=data_in_range, orient='v', palette="Set2")
plt.ylabel('Value', fontsize=20)
plt.xlabel('CellType', fontsize=20)
plt.title('Random sampling data distribution', fontsize=25)
plt.xticks(rotation=45, fontsize=15)
plt.yticks(fontsize=15)

# 调整布局以适应标签
plt.tight_layout()
# 保存图形为PNG文件
plt.savefig(data_path + '_boxplot_distribution.png')

# 显示图形
plt.show()



