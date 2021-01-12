import pandas as pd
import matplotlib.pyplot as plt
import warnings

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')  # 忽略警告

df = pd.read_csv('training_data.csv')

new_col = ['Date', 'Time', 'Cell_num', 'Upload', 'Download']
df.columns = new_col
df = df.drop_duplicates(keep='first')
df['Time'] = df['Time'].apply(lambda x: int(x.split(':')[0]))  # 对时间进行处理


# def date_trans(df):  # 对日期进行处理
#     date = str(df['Date'])
#     if '-' in date:
#         temp = pd.datetime.strptime("2" + date, "%Y-%m-%d")
#     else:
#         temp = pd.datetime.strptime(date, "%Y/%m/%d")
#     return temp.strftime('%Y/%m/%d')
#
#
# df['Date'] = df.apply(date_trans, axis=1)
# df['Mon'] = df['Date'].apply(lambda x: int(x.split('/')[0]))
# df['Day'] = df['Date'].apply(lambda x: int(x.split('/')[2]))
df['Mon'] = df['Date'].apply(lambda x: int(x[5]))
df['Day'] = df['Date'].apply(lambda x: int(x.split('/')[-1][-2:]))


# 分组排序
df = df.groupby(['Cell_num', 'Mon', 'Day'], sort=True).apply(lambda x: x.sort_values("Time", ascending=True)).reset_index(
    drop=True)
df.to_csv('基站日期_时间排序.csv', index=False)
# 筛选休息日
weekday_3 = [3, 4, 10, 11, 17, 18, 24, 25, 31]
weekday_4 = [1, 7, 8, 14, 15]
week = df[((df["Mon"] == 3) & (df["Day"].isin(weekday_3))) | ((df["Mon"] == 4) & (df["Day"].isin(weekday_4)))]
week.to_csv('周末排序.csv', index=False)
work = df[((df["Mon"] == 3) & (~df["Day"].isin(weekday_3))) | ((df["Mon"] == 4) & (~df["Day"].isin(weekday_4)))]
work.to_csv('工作日排序.csv', index=False)



