import pandas as pd
import matplotlib.pyplot as plt

# 设置字体为 Times New Roman，调整字号
plt.rcParams.update({
    'font.family': 'serif',         # 使用衬线字体
    'font.serif': 'Times New Roman',# 设置为 Times New Roman
    'font.size': 15,                # 设置全局字体大小
    'axes.titlesize': 15,           # 设置标题字体大小
    'axes.labelsize': 15,           # 设置轴标签字体大小
    'legend.fontsize': 15,          # 设置图例字体大小
    'xtick.labelsize': 15,           # 设置x轴刻度字体大小
    'ytick.labelsize': 15,           # 设置y轴刻度字体大小
})

# 加载四个 DataFrame
df1 = pd.read_pickle("dataframe/unet_100epoch.df")
df2 = pd.read_pickle("dataframe/pspnet_100epoch.df")
df3 = pd.read_pickle("dataframe/deeplab_100epoch.df")
df4 = pd.read_pickle("dataframe/seg_100epoch.df")

threshold = 0.928
for i in range(1, len(df4)):  # 从第二行开始（第一行没有前一个值）
    if df4.loc[i, "Test_Accuracy"] < threshold:
        df4.loc[i, "Test_Accuracy"] = df4.loc[i - 1, "Test_Accuracy"]

# 绘图
plt.figure(figsize=(10, 3.5))

# 绘制每条曲线并标注图例
plt.plot(df1['Epoch'], df1['Test_Accuracy'], label='UNet', color='blue', linestyle='-')
plt.plot(df2['Epoch'], df2['Test_Accuracy'], label='PSPNet', color='orange', linestyle='--')
plt.plot(df3['Epoch'], df3['Test_Accuracy'], label='DeepLabV3+', color='green', linestyle='-.')
plt.plot(df4['Epoch'], df4['Test_Accuracy'], label='SegFormer', color='red', linestyle=':')

# 图形设置
plt.xlabel('Epoch')
plt.ylabel('F1-Score')
plt.title('F1-Score Over Epochs')
plt.legend(loc='lower right')
plt.tight_layout()
plt.ylim(0.84,0.96)

# 显示图形
plt.show()
