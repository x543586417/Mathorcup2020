import math
import matplotlib.pyplot as plt
import warnings
import pandas as pd

plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')  # 忽略警告

#65288 9199 9217 10126 43933

df = pd.read_csv(r'/Users/lucifer/Documents/Competition/Mathorcup2020-RuiXing/处理后数据/附件2：短期验证选择的小区数据集.csv', encoding='gbk')

new_col = ['DATE','Hour','Cell_num' , 'Upload','Download']
df.columns = new_col

b=df.loc[:,'Cell_num'].value_counts()

