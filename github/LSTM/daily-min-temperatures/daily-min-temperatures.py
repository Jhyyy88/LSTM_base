import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

#读取数据
data = pd.read_csv(r"D:\Pycharm\Torch_Project\github\数据集\daily-min-temperatures.csv")
data = data['Temp']
dataset = data.values
dataset = dataset.astype('float32')

#数据归一化
max_value = np.max(dataset)
min_value = np.min(dataset)
scalar = max_value - min_value
dataset = list(map(lambda x: (x - min_value) / scalar,dataset))

#创建数据集
def create_dataset(dataset,step = 2):
    data_x,data_y = [],[]
    for i in range(len(dataset) - step):
        a = dataset[i:i+step]
        b = dataset[i+step]
        data_x.append(a)
        data_y.append(b)
    return np.array(data_x),np.array(data_y)

#划分训练集与测试集
dataX,dataY = create_dataset(dataset)
train_split = int(len(dataX) * 0.7)
trainX = dataX[:train_split]
trainY = dataY[:train_split]
testX = dataX[train_split:]
testY = dataY[train_split:]

trainX = trainX.reshape(-1,1,2)
trainY = trainY.reshape(-1,1,1)
testX = testX.reshape(-1,1,2)
testY = testY.reshape(-1,1,1)

trainX = torch.from_numpy(trainX)
trainY = torch.from_numpy(trainY)
testX = torch.from_numpy(testX)
testY = torch.from_numpy(testY)

#构建神经网络
class Net_temperature(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,num_layers):
        super(Net_temperature,self).__init__()
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers)
        self.linear = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        x,_ = self.lstm(x)
        s,b,h = x.shape
        x = x.view(s*b,h)
        x = self.linear(x)
        out = x.view(s,b,-1)
        return out

#开始训练
net = Net_temperature(input_size=2,hidden_size=50,output_size=1,num_layers=2)
lr = 0.01
optim = torch.optim.Adam(net.parameters(),lr=lr)
loss_func = nn.MSELoss()

global_step = 0
epochs = 1000
net.train()
print('=====训练开始=====')
for i in range(epochs):
    output = net(trainX)
    loss = loss_func(output,trainY)
    optim.zero_grad()
    loss.backward()
    optim.step()
    global_step += 1
    if(global_step % 100 == 0):
        print('第{}轮训练的损失为：{}'.format(global_step,loss))
print('=====训练结束=====')
torch.save(net,'finish_net')

net.eval()
pred_X = net(testX)
pred_X = pred_X.view(-1).data.numpy()*scalar + min_value
testY = testY.view(-1).data.numpy()*scalar + min_value
plt.plot(pred_X,label='PRED')
plt.plot(testY,label='REAL')
plt.legend(loc='best')
plt.show()





