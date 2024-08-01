import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_heatmap(file_path, name):
    # 读取文本文件
    data = pd.read_csv(file_path, sep='\s+', header=None, names=['Length', 'Size', 'Time'])
    
    # 将所有数据转换为数值类型，非数值数据会被转换为 NaN
    data['Length'] = pd.to_numeric(data['Length'], errors='coerce')
    data['Size'] = pd.to_numeric(data['Size'], errors='coerce')
    data['Time'] = pd.to_numeric(data['Time'], errors='coerce')

    # 移除任何包含 NaN 的行（即非数值数据）
    data = data.dropna()

    # 获取所有不同的序列长度和模型大小
    lengths = data['Length'].unique()
    sizes = data['Size'].unique()
    
    # 创建一个空矩阵来存储时间数据
    matrix = pd.DataFrame(index=lengths, columns=sizes)
    
    # 填充矩阵
    for index, row in data.iterrows():
        length = row['Length']
        size = row['Size']
        time = row['Time']
        if 'prefill' in name:
            matrix.at[length, size] = time / 1000
        else:
            matrix.at[length, size] = time
    
    # 打印 matrix 查看其内容
    print(matrix)
    
    # 检查 matrix 的数据类型
    print("Data Types:")
    print(matrix.dtypes)
    
    # 强制转换 matrix 的数据类型为 float
    matrix = matrix.astype(float)
    
    # 再次检查 matrix 的数据类型
    print("Data Types after conversion:")
    print(matrix.dtypes)
    
    # 反转 y 轴顺序
    matrix = matrix.reindex(index=reversed(matrix.index))
    
    # 绘制热力图
    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(matrix, annot=False, fmt=".1f", cmap='Reds')  # 改变颜色调为红色系
    # plt.title('Inference Time Heatmap')
    # 设置 x 轴和 y 轴的刻度标签字体大小
    plt.xticks(fontsize=20)  # 设置 x 轴刻度标签字体大小
    plt.yticks(fontsize=20, rotation=30)  # 设置 y 轴刻度标签字体大小

    # 获取 colorbar 对象
    cbar = heatmap.collections[0].colorbar
    
    # 设置 colorbar 标签
    if 'prefill' in name:
        cbar.set_label('TTFT (sec.)', fontsize=26)  # 设置字体大小
    else:
        cbar.set_label('TPOT (ms)', fontsize=26)  # 设置字体大小
    cbar.ax.tick_params(labelsize=20)
    plt.xlabel('Model Size', fontsize=26)
    plt.ylabel('Prompt Length', fontsize=26)
    plt.show()
    plt.savefig(name, format='pdf')

# 调用函数
plot_heatmap('scripts/prefill_llama7b.txt', 'heatmap_prefill.pdf')
plot_heatmap('scripts/decode_llama7b.txt', 'heatmap_decode.pdf')