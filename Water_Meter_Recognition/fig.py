import pandas as pd
import matplotlib.pyplot as plt

# Configure global font settings for publication-quality figures
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

# Load evaluation results
df1 = pd.read_pickle("dataframe/cnn_100epoch.df")
df2 = pd.read_pickle("dataframe/vgg_100epoch.df")
df3 = pd.read_pickle("dataframe/res_100epoch.df")
df4 = pd.read_pickle("dataframe/den_100epoch.df")

plt.figure(figsize=(10, 3.5))

# Plot test accuracy curves of different models
plt.plot(df1['Epoch'], df1['Test_Accuracy'], label='CNN', linestyle='-')
plt.plot(df2['Epoch'], df2['Test_Accuracy'], label='VGG16', linestyle='--')
plt.plot(df3['Epoch'], df3['Test_Accuracy'], label='ResNet', linestyle='-.')
plt.plot(df4['Epoch'], df4['Test_Accuracy'], label='DenseNet', linestyle=':')

plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.title('Recognition test accuracy over epochs')
plt.legend(loc='best')
plt.tight_layout()

plt.show()
