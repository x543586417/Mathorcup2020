import matplotlib.pyplot as plt
import warnings
import pandas as pd
from pandas import DataFrame

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')  # 忽略警告

#65288 9199 9217 10126 43933

df = pd.read_csv('长期每日上下行流量和.csv', encoding='gbk')
a = df[df['Cell_num'] == 9199]['Download'].tolist()
plt.plot(a, 'royalblue')
plt.show()


