import pandas as pd
import matplotlib.pyplot as plt
import warnings
import re
import time

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')  # 忽略警告

df = pd.read_csv('training_data.csv', encoding='gbk')
df_short = pd.read_csv('清理后数据/附件2：短期验证选择的小区数据集.csv', encoding='gbk')
df_long = pd.read_csv('清理后数据/附件3：长期验证选择的小区数据集.csv', encoding='gbk')

new_col = ['Date', 'Time', 'Cell_num', 'Upload', 'Download']
df.columns = new_col

df = df.drop_duplicates(keep='first')
df['Time'] = df['Time'].apply(lambda x: int(x.split(':')[0]))  # 对时间进行处理

#  日期序号表示成 日+（月-3）*31
df['Day'] = df['Date'].apply(lambda x: int((re.split('[-/]', x))[2]) + (int(x[5]) - 3) * 31)

a = df_short['cell_num'].unique()
#  筛选长短期需要基站数据集
df_s = df[df['Cell_num'].isin(df_short['cell_num'].unique())]
df_l = df[df['Cell_num'].isin(df_long['小区编号'].unique())]

# ---------短期筛选排序----------#
# 排序
df = df_s.sort_values(by=['Cell_num', 'Day', 'Time'], ascending=True)
df.to_csv('短期基站日期_时间排序.csv', index=False)
# 筛选休息日
weekday = [3, 4, 10, 11, 17, 18, 24, 25, 31, 32, 38, 39, 45, 46]
week = df[df["Day"].isin(weekday)]
week.to_csv('短期周末排序.csv', index=False)
work = df[~df["Day"].isin(weekday)]
work.to_csv('短期工作日排序.csv', index=False)

# ----------长期筛选排序---------#
# 排序
df = df_l.sort_values(by=['Cell_num', 'Day', 'Time'], ascending=True)
df.to_csv('长期基站日期_时间排序.csv', index=False)
# 筛选休息日
weekday = [3, 4, 10, 11, 17, 18, 24, 25, 31, 32, 38, 39, 45, 46]
week = df[df["Day"].isin(weekday)]
week.to_csv('长期周末排序.csv', index=False)
work = df[~df["Day"].isin(weekday)]
work.to_csv('长期工作日排序.csv', index=False)
