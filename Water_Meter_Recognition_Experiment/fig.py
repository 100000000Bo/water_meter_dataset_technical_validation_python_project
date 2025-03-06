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
df1 = pd.read_pickle("dataframe/cnn_100epoch.df")
df2 = pd.read_pickle("dataframe/vgg_100epoch.df")
df3 = pd.read_pickle("dataframe/res_100epoch.df")
df4 = pd.read_pickle("dataframe/den_100epoch.df")

# 绘图
plt.figure(figsize=(10, 3.5))

# 绘制每条曲线并标注图例
plt.plot(df1['Epoch'], df1['Test_Accuracy'], label='CNN', color='blue', linestyle='-')
plt.plot(df2['Epoch'], df2['Test_Accuracy'], label='VGG16', color='orange', linestyle='--')
plt.plot(df3['Epoch'], df3['Test_Accuracy'], label='ResNet', color='green', linestyle='-.')
plt.plot(df4['Epoch'], df4['Test_Accuracy'], label='DenseNet', color='red', linestyle=':')

# 图形设置
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy Over Epochs')
plt.legend(loc='best')
plt.tight_layout()

# 显示图形
plt.show()
