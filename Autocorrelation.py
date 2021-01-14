import numpy as np
import matplotlib.pyplot as plt


#法1
def autocorrelation(x, lags):
    # 计算lags阶以内的自相关系数，返回lags个值，分别计算序列均值，标准差
    n = len(x)
    x = np.array(x)
    result = [np.correlate(x[i:] - x[i:].mean(), x[:n - i] - x[:n - i].mean())[0] \
              / (x[i:].std() * x[:n - i].std() * (n - i)) for i in range(1, lags + 1)]
    return result

#法2
y = train['quantity']
x = np.arange(len(train))
yunbiased = y-np.mean(y)
ynorm = np.sum(yunbiased**2)
acor = np.correlate(yunbiased, yunbiased, "same")/ynorm
# # use only second half
acor = acor[int(len(acor)/2):]
fig = plt.figure(figsize = (10,6))
plt.xlabel('Lag', fontsize=15)
plt.ylabel('ACF', fontsize=15)
plt.plot(acor)
plt.show()

