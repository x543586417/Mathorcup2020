from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import pandas as pd

# 数据集
df = pd.read_csv('处理后数据/短期工作日排序.csv')
# Calculate ACF and PACF upto 50 lags# acf_50 = acf(df.value, nlags=50)# pacf_50 = pacf(df.value, nlags=50)
# Draw Plot
for i in [186, 420, 5229, 9609]:
    fig, axes = plt.subplots(2, 1, figsize=(8, 4), dpi=100)
    plot_acf(df[df['Cell_num'] == i]['Download'].tolist(), lags=50, ax=axes[0])
    plot_pacf(df[df['Cell_num'] == i]['Download'].tolist(), lags=50, ax=axes[1])
    plt.show()
