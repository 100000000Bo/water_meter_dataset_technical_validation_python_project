import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': 'Times New Roman',
    'font.size': 15,
    'axes.titlesize': 15,
    'axes.labelsize': 15,
    'legend.fontsize': 15,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
})

# Load the test result DataFrame of the four models
df1 = pd.read_pickle("dataframe/unet_100epoch.df")
df2 = pd.read_pickle("dataframe/pspnet_100epoch.df")
df3 = pd.read_pickle("dataframe/deeplab_100epoch.df")
df4 = pd.read_pickle("dataframe/seg_100epoch.df")

plt.figure(figsize=(10, 3.5))

# Draw each curve
plt.plot(df1['Epoch'], df1['Test_Accuracy'], label='UNet', color='blue', linestyle='-')
plt.plot(df2['Epoch'], df2['Test_Accuracy'], label='PSPNet', color='orange', linestyle='--')
plt.plot(df3['Epoch'], df3['Test_Accuracy'], label='DeepLabV3+', color='green', linestyle='-.')
plt.plot(df4['Epoch'], df4['Test_Accuracy'], label='SegFormer', color='red', linestyle=':')

plt.xlabel('Epoch')
plt.ylabel('F1-Score')
plt.title('Segmentation F1-Score over epochs')
plt.legend(loc='lower right')
plt.tight_layout()
plt.ylim(0.84,0.96)

plt.show()
