import pandas as pd
import numpy as np
import math
import keras
from matplotlib import pyplot as plt
from matplotlib.pylab import mpl
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K
from keras.layers import LeakyReLU
from sklearn.metrics import mean_squared_error # 均方误差
from keras.callbacks import LearningRateScheduler
from keras.callbacks import EarlyStopping
from keras import Input, Model,Sequential
from keras.layers import Bidirectional#, Concatenate
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import optimizers

#转成有监督数据
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    #数据序列(也将就是input) input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        #预测数据（input对应的输出值） forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    #拼接 put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # 删除值为NAN的行 drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


##数据预处理 load dataset
dataset = read_csv('data/los_for_model.csv', header=0, index_col=0)
values = dataset.values
#标签编码 integer encode direction
encoder = LabelEncoder()
values[:, 1] = encoder.fit_transform(values[:, 1])
values[:, 2] = encoder.fit_transform(values[:, 2])
#保证为float ensure all data is float
values = values.astype('float32')
#归一化 normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
#转成有监督数据 frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
#删除不预测的列 drop columns we don't want to predict
reframed.drop(reframed.columns[[34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65]], axis=1, inplace=True)
print(reframed.head())

#数据准备
#把数据分为训练数据和测试数据 split into train and test sets
values = reframed.values
#拿一年的时间长度训练
n_train = 290
#划分训练数据和测试数据
train = values[:n_train, :] # 第一年的数据作为训练数据
test = values[n_train:, :] # 后三年的数据作为测试数据
#拆分输入输出 split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
#reshape输入为LSTM的输入格式 reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print ('train_x.shape, train_y.shape, test_x.shape, test_y.shape')
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# def scheduler(epoch):
#     # 每隔50个epoch，学习率减小为原来的1/10
#     if epoch % 50 == 0 and epoch != 0:
#         lr = K.get_value(bilstm.optimizer.lr)
#         if lr>1e-5:
#             K.set_value(bilstm.optimizer.lr, lr * 0.1)
#             print("lr changed to {}".format(lr * 0.1))
#     return K.get_value(bilstm.optimizer.lr)

# reduce_lr = LearningRateScheduler(scheduler)
# early_stopping = EarlyStopping(monitor='loss',
#                                patience=20,
#                                min_delta=1e-5,
#                                mode='auto',
#                                restore_best_weights=False,#是否从具有监测数量的最佳值的时期恢复模型权重
#                                verbose=2)


adam = optimizers.Adam(learning_rate=0.0001)

# 特征数
input_size = train_X.shape[2]
# 时间步长：用多少个时间步的数据来预测下一个时刻的值
time_steps = train_X.shape[1]
# 隐藏层block的个数
batch_size=128

bilstm = keras.Sequential()
bilstm.add(Bidirectional(keras.layers.LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])), merge_mode='concat'))
bilstm.add(keras.layers.Dense(1))
bilstm.compile(optimizer=adam, loss='mae', metrics=['accuracy'])

history=bilstm.fit(train_X,train_y,validation_data=(test_X,test_y), epochs=7000,batch_size=128,shuffle=False)

#输出 plot history
pyplot.plot(history.history['loss'], label='BiLSTM_train')
pyplot.plot(history.history['val_loss'], label='BiLSTM_test')
pyplot.legend()
pyplot.show()

#进行预测 make a prediction
yhat = bilstm.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
#预测数据逆缩放 invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
inv_yhat = np.array(inv_yhat)
#真实数据逆缩放 invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]

#画出真实数据和预测数据
pyplot.plot(inv_yhat,label='BiLSTM_prediction')
pyplot.plot(inv_y,label='BiLSTM_true')
pyplot.legend()
pyplot.show()

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
