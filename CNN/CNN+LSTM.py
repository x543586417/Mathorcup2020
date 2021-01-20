import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Convolution1D, MaxPooling1D
from keras.layers import LSTM



# 设置网络参数
BATCH_SIZE = 24
HIDDEN_SIZE = 128


# 设置数据集(考虑滑动窗口即batch_size，所以新构造的数据集长度必定为原长-batch_size
#           ，然后这边X,Y分别是输出值，输出值，根据X来求Y，因为是训练集，所以Y告诉你)
def create_dataset(dataset, loop_back=BATCH_SIZE):
    dataX, dataY = [], []
    for i in range(len(dataset) - loop_back):
        a = dataset[i:(i + loop_back)]  # 这边的话，因为切片是不包含的末尾位的，所以a是两个数据组成
        dataX.append(a)
        dataY.append(dataset[i + loop_back])
    return np.array(dataX), np.array(dataY)



#模型参数
time_step=100
# Convolution  卷积
filter_length = 5    # 滤波器长度
nb_filter = 64       # 滤波器个数
pool_length = 4      # 池化长度
# LSTM
lstm_output_size = 70   # LSTM 层输出尺寸
# Training   训练参数
batch_size = 30   # 批数据量大小
nb_epoch = 2      # 迭代次数
# 构建模型
model = Sequential()
model.add(input(shape=(time_step, 128))) # 输入特征接收维度)  # 词嵌入层
model.add(Dropout(0.25))       # Dropout层
# 1D 卷积层，对词嵌入层输出做卷积操作
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
# 池化层
model.add(MaxPooling1D(pool_length=pool_length))
# LSTM 循环层
model.add(LSTM(lstm_output_size))
# 全连接层，只有一个神经元，输入是否为正面情感值
model.add(Dense(1))
model.add(Activation('sigmoid'))  # sigmoid判断情感（此处来做文本的情感分类问题）
model.summary()   # 模型概述
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# 训练
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          validation_data=(X_test, y_test))

