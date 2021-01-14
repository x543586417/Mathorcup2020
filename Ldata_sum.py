import pandas as pd
import matplotlib.pyplot as plt
import warnings

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')  # 忽略警告

df_sort = pd.read_csv('处理后数据/长期基站日期_时间排序.csv', encoding='gbk')
df_work = pd.read_csv('处理后数据/长期工作日排序.csv', encoding='gbk')
df_week = pd.read_csv('处理后数据/长期周末排序.csv', encoding='gbk')

# ----------根据基站日期对上下行流量求和---------#
# 排序
df_sort = df_sort.drop(labels='Date', axis=1)
df = df_sort.groupby(['Cell_num', 'Day'])['Upload', 'Download'].sum().reset_index()
df.to_csv('长期每日上下行流量和.csv', index=False)

df_work = df_work.drop(labels='Date', axis=1)
df = df_work.groupby(['Cell_num', 'Day'])['Upload', 'Download'].sum().reset_index()
df.to_csv('长期工作日上下行流量和.csv', index=False)

df_week = df_week.drop(labels='Date', axis=1)
df = df_week.groupby(['Cell_num', 'Day'])['Upload', 'Download'].sum().reset_index()
df.to_csv('长期周末上下行流量和.csv', index=False)

