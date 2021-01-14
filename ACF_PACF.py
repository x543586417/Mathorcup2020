from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import pandas as pd
#数据集
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv')
# Calculate ACF and PACF upto 50 lags# acf_50 = acf(df.value, nlags=50)# pacf_50 = pacf(df.value, nlags=50)
# Draw Plot
fig, axes = plt.subplots(1, 2, figsize=(16, 3), dpi=100)
plot_acf(df.value.tolist(), lags=50, ax=axes[0])
plot_pacf(df.value.tolist(), lags=50, ax=axes[1])
plt.show()
