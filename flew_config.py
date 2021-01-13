import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')  # 忽略警告

df = pd.read_csv('周末排序.csv')
cell_n = df["Cell_num"].nunique()  # 基站数目
Day_n = 50  # 天数总共为50
up_tr_day = np.zeros((cell_n, Day_n))
down_tr_day = np.zeros((cell_n, Day_n))
upload = []  # 连续时间的列表
download = []

# for i in range(1, cell_n + 1):
#     temp_up = df['Upload'][df['Cell_num'] == i].tolist()
#     temp_down = df['Download'][df['Cell_num'] == i].tolist()
#
#     upload.append(temp_up)
#     download.append(temp_down)
for i in df['Cell_num'].unique():   # 每个基站的流量转换为list（基站编号从1开始）
    temp_up = df[df['Cell_num'] == i]
    temp_down = df[df['Cell_num'] == i]
    upload.append(temp_up['Upload'].tolist())
    download.append(temp_down['Download'].tolist())
    for j in df['Day'].unique():
        temp_up_day = temp_up[temp_up['Day'] == j]
        temp_down_day = temp_down[temp_down['Day'] == j]
        if i == 3:  # 这里可以改对哪个基站，下面可以改上下行链路
            plt.plot(np.array(temp_up_day['Download'].tolist()))
plt.show()

# 这里是对某个基站长期的上下行链路数据作图
# for i in range(cell_n):
#     plt.plot(download[i])
# plt.show()
# plt.figure(1,2,1)
# plt.plot(upload[1])
# plt.plot(download[2])

