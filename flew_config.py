import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')  # 忽略警告

df = pd.read_csv('基站日期_时间排序.csv')

temp = []
for i in df['Cell_num']:
    if i not in temp:
        temp.append(i)
cell_n = len(temp)

upload = []
download = []
for i in range(1, cell_n + 1):  # 每个基站的流量转换为list（基站编号从1开始）
    temp_up = df['Upload'][df['Cell_num'] == i].tolist()
    temp_down = df['Download'][df['Cell_num'] == i].tolist()
    upload.append(temp_up)
    download.append(temp_down)

# for i in range(cell_n):
#     plt.plot(upload[i])
# plt.figure(1,2,1)
# plt.plot(upload[1])
plt.plot(download[1])
plt.show()
