import pandas as pd
import matplotlib.pyplot as plt

# Configure global font to Times New Roman
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

# Load data
df1 = pd.read_pickle("dataframe/blurry.df")
df2 = pd.read_pickle("dataframe/dark.df")
df3 = pd.read_pickle("dataframe/dial-stained.df")
df4 = pd.read_pickle("dataframe/soil-covered.df")
df5 = pd.read_pickle("dataframe/reflective.df")

# Plot F1-Score over epochs
plt.figure(figsize=(10, 5))
plt.plot(df1['Epoch'], df1['F1-score'], label='blurry', color='blue')
plt.plot(df2['Epoch'], df2['F1-score'], label='dark', color='orange')
plt.plot(df3['Epoch'], df3['F1-score'], label='dial-stained', color='green')
plt.plot(df4['Epoch'], df4['F1-score'], label='soil-covered', color='red')
plt.plot(df5['Epoch'], df5['F1-score'], label='reflective', color='purple')

# Set labels, title, legend, and layout
plt.xlabel('Epoch')
plt.ylabel('F1-Score')
plt.title('F1-Score Over Epochs')
plt.legend(loc='lower right')
plt.ylim(0.4, 1)
plt.tight_layout()

# Display the plot
plt.show()
