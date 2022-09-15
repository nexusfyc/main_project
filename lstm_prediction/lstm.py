import torch

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import register_matplotlib_converters
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import random
from scipy.interpolate import UnivariateSpline  
import pickle
from sklearn.decomposition import PCA
import time
import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt


class MV_LSTM(torch.nn.Module):
    def __init__(self,n_features,seq_length):
        super(MV_LSTM, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.n_hidden = 30 # number of hidden states
        self.n_layers = 4 # number of LSTM layers (stacked)
        self.dropout = nn.Dropout(0.1) 

        self.l_lstm = torch.nn.LSTM(input_size = n_features, 
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers, 
                                 batch_first = True
                                 )
        # according to pytorch docs LSTM output is 
        # (batch_size,seq_len, num_directions * hidden_size)
        # when considering batch_first = True
        self.l_linear = torch.nn.Linear(self.n_hidden*self.seq_len, 100)
        self.sigmoid = nn.Sigmoid()
     
        


    def init_hidden(self, batch_size):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(self.n_layers,batch_size,self.n_hidden)
        cell_state = torch.zeros(self.n_layers,batch_size,self.n_hidden)
        self.hidden = (hidden_state, cell_state)


    def forward(self, x):        
        batch_size, seq_len, _ = x.size()

        lstm_out, self.hidden = self.l_lstm(x,self.hidden)
        #lstm_out, self.hidden = self.l_lstm(x)
        
        # lstm_out(with batch_first = True) is 
        # (batch_size,seq_len,num_directions * hidden_size)
        # for following linear layer we want to keep batch_size dimension and merge rest       
        # .contiguous() -> solves tensor compatibility error
        x = lstm_out.contiguous().view(batch_size,-1)
        return self.sigmoid(self.l_linear(x))
    
from numpy import array
from numpy import hstack

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    
    for i in range(0,len(sequences),100):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if i!=0 and end_ix > len(sequences):
            break
        
        sequences[i:end_ix,0]=np.insert(np.diff(sequences[i:end_ix,0]),0,0)
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix-33], sequences[end_ix-33:end_ix]
        
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

#read training data#############################################################################################
df = pd.read_csv('../data/time-series-19-covid-combined.csv', skiprows=1)
df.head()
df.info()
df.columns = ['date','country', 'province','confirmed','recovered','deaths']

is_china =  (df['country']=='China')

#  read testing data##############################################################################################
df2 = pd.read_csv('../data/time-series-19-covid-combined.csv', skiprows=1)
df2.head()
df2.info()
df2.columns = ['date','country', 'province', 'confirmed','recovered','deaths']

is_indonesia =  (df2['country']=='Indonesia')

#  training data filtering#########################################################################################
#  模型训练数据
data=df[df.country.isin(['China','Germany','Australia','Brazil','US','Belgium','Spain','Italy','UK','France','Japan','Malaysia','Vietnam','Iran','UEA','Singapore','Thailand','Korea, South','Japan','Iran','Netherlands','Russia','Chile','India','Greece','Mexico','Mongolia','Philippines','New Zealand','South Africa','Botswana','Uruguay','Paraguay','Madagascar','Peru', 'Portugal', 'Denmark','Hungary','Kenya','Ireland','Israel','Norway','Mauritius','Rwanda','Iceland','Kazakhstan','Switzerland','Cyprus','Zimbabwe'])][['confirmed','recovered','deaths']]


#  testing data filtering#########################################################################################
#  印度尼西亚数据
data2=df2[(is_indonesia)][['confirmed','recovered','deaths']]
#  印尼日期和确诊数据
date=df2[(is_indonesia)][['date','confirmed']]

date.day = pd.to_datetime(date.date,format='%Y%m%d', errors='ignore')
date.set_index('date', inplace=True)

print('############test##############')
df3 = pd.read_csv('../data/Indonesia.csv', skiprows=1)
df3.head()
df3.info()
df3.columns = ['date','country', 'province', 'confirmed','recovered','deaths']

is_indonesia =  (df3['country']=='Indonesia')
date_indo=df3[(is_indonesia)][['date','confirmed']]
date_indo.day = pd.to_datetime(date_indo.date,format='%Y%m%d', errors='ignore')
date_indo.set_index('date', inplace=True)
print('############test##############')
################################################################################################################

n_features = 3 # this is number of parallel inputs
n_timesteps = 100 # this is number of timesteps

#  input splitting################################################################################################
# X中的数据为：每100个为一组，一组中的前67个；Y中的数据为：每100个为一组，一组中的后33个
X, Y = split_sequences(data.values, n_timesteps)

#  打印X、Y维度：X为(840,67,3) Y为(840,33,3)
print (X.shape,Y.shape)
#  normalization##################################################################################################
# 将X和Y拼接
alld=np.concatenate((X,Y),1)
print("alld形状为：")
print(alld.shape) # alld(840,100,3)
alld=alld.reshape(alld.shape[0]*alld.shape[1],alld.shape[2]) # (84000,3)

# 数据归一化：将数据映射入0-1的区间内
scaler = MinMaxScaler()
scaler.fit(alld)
X=[scaler.transform(x) for x in X]
y=[scaler.transform(y) for y in Y]

X=np.array(X)
print(X.shape)
y=np.array(y)[:,:,0]  # 只取出确诊的数据（1：确诊 2：治愈 3：死亡）


#  training#########################################################################################

mv_net = MV_LSTM(n_features,67) # seq_length = 67
criterion = torch.nn.MSELoss() # reduction='sum' created huge loss value
optimizer = torch.optim.Adam(mv_net.parameters(), lr=1e-3)

train_episodes = 10

batch_size = 16

mv_net.train() # 将模型设置为训练模式

for t in range(train_episodes):
    
    for b in range(0,len(X),batch_size):
       
        p = np.random.permutation(len(X)) # 生成一个0-839的随机排列array
        
        inpt = X[p][b:b+batch_size,:,:] # 随机从X(840,67,3)的840个样本中选出16个
        target = y[p][b:b+batch_size,:]    # 随机从y(840,33)的840个样本中选出16个

        
        x_batch = torch.tensor(inpt,dtype=torch.float32) # 将输入inpt转换为tensor(16,67,3)
        y_batch = torch.tensor(target,dtype=torch.float32) # 将输出target转换为tensor(16,33)

        mv_net.init_hidden(x_batch.size(0)) # x_batch.size(0) = 16 初始化h0
        
        output = mv_net(x_batch) # output(16,100)
        all_batch=torch.cat((x_batch[:,:,0], y_batch), 1) # 将第一个参数中的数组按列拼接，拼接后all_batch(16,100)
        
        
        loss = 1000*criterion(output.view(-1), all_batch.view(-1))  

        loss.backward()
        optimizer.step()        
        optimizer.zero_grad() 
    print('step : ' , t , 'loss : ' , loss.item())


#evaluation#########################################################################################################
#data2x=data2[~(data2.confirmed==0)]
data2x=data2 # 印度尼西亚的确诊、康复、死亡数据
truth = data2 #  同上
print("没有归一化的数据：")
print(data2x)
#将印度尼西亚的确诊数据（第一列）改为每日新增数据
data2x.values[0:len(data2x),0]=np.insert(np.diff(data2x.values[0:len(data2x),0]),0,0)
print('没有归一化的数据（第一列修改为每日新增）:',data2x)

print('************test************')
scaler.fit(data2x)
print('************test************')

data2x=scaler.transform(data2x)
print('归一化后的数据：',data2x)

X_test = np.expand_dims(data2x, axis=0) # 在第一个位置新增一个维度,原data2x(816,3)
print (X_test.shape) # X_test(1,816,3)
mv_net.init_hidden(1)
lstm_out = mv_net(torch.tensor(X_test[:, -67:, :], dtype=torch.float32)) # 将X_test最后67组样本输入模型
print('lstm_out_shape:', lstm_out.shape)
lstm_out=lstm_out.reshape(1,100,1).cpu().data.numpy()

print (data2x[-67:,0],lstm_out)
# inverse_transform方法将归一化的数据还原
actual_predictions = scaler.inverse_transform(np.tile(lstm_out, (1, 1, 3))[0])[:,0]

# print('*************test***************')
#
# new_actual_prediction = scaler.inverse_transform(lstm_out)
# print('爆改预测值：',new_actual_prediction)
# print('*************test***************')
# print('data2后33个数据和预测结果为：')
print (data2.values[-67:,0],actual_predictions)

#actual_predictions=lstm_out

x = np.arange(0, 54, 1)
x2 = np.arange(0, 67, 1)
x3 = np.arange(0, 100, 10)
x4 = np.arange(0, 50, 1)

#save prediction
with open('../data/predict_indo8.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(pd.Series(actual_predictions), f,protocol=2)

#visualization####################################################################################################    
fig, ax = plt.subplots() 
plt.title('Days vs Confirmed Cases Accumulation')
plt.ylabel('Confirmed')

left, width = .25, .5
bottom, height = .25, .5
right = left + width
top = bottom + height

print (date.index)
date_list=pd.date_range(start=date.index[0],end=date.index[99])

date_list2=pd.date_range(start=date.index[0],end=date.index[-1])

print (date_list)

plt.axvline(x=np.array(date_list)[66], color='r', linestyle='--')

ax.text(0.2*(left+right), 0.8*(bottom+top), 'input sequence',
        horizontalalignment='left',
        verticalalignment='center',
        fontsize=10, color='red',
        transform=ax.transAxes)
ax.text(0.0125*(left+right), 0.77*(bottom+top), '______________________',
        horizontalalignment='left',
        verticalalignment='center',
        fontsize=20, color='red',
        transform=ax.transAxes)

sumpred = np.cumsum(np.absolute(actual_predictions))

print (date.values.shape) 
# print (sqrt(mean_squared_error(date.confirmed, sumpred)))
#  plt.plot(date.values[-67:],np.cumsum(data2.confirmed.values[-67:]))
plt.plot(np.array(date_list),sumpred,label='Prediction')
plt.plot(np.array(date_list),date_indo.confirmed,label='Actual')
plt.xticks(rotation=90)
fig.autofmt_xdate()
plt.legend(loc=2)
plt.show()