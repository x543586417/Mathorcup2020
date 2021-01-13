import pandas as pd
import matplotlib.pyplot as plt
import warnings
import re
import time

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')  # 忽略警告

df = pd.read_csv('training_data.csv')

new_col = ['Date', 'Time', 'Cell_num', 'Upload', 'Download']
df.columns = new_col

df = df.drop_duplicates(keep='first')
df['Time'] = df['Time'].apply(lambda x: int(x.split(':')[0]))  # 对时间进行处理

#  日期序号表示成 日+（月-3）*31
df['Day'] = df['Date'].apply(lambda x: int((re.split('[-/]', x))[2]) + (int(x[5]) - 3) * 31)
# 排序
time_start = time.time()
df = df.sort_values(by=['Cell_num', 'Day', 'Time'], ascending=True)
time_end = time.time()
print('totally cost', time_end - time_start)
df.to_csv('基站日期_时间排序.csv', index=False)

# 筛选休息日
weekday = [3, 4, 10, 11, 17, 18, 24, 25, 31, 32, 38, 39, 45, 46]
week = df[df["Day"].isin(weekday)]
week.to_csv('周末排序.csv', index=False)
work = df[~df["Day"].isin(weekday)]
work.to_csv('工作日排序.csv', index=False)
