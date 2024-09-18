import matplotlib.pyplot as plt

from data import pre_process
import torch
from torch import nn
import numpy as np
#获取数据
trainX, trainY, testX, testY, max, min = pre_process(r'D:\Pycharm\Torch_Project\github\数据集\shampoo-sales.csv')

#构建神经网
class net_shampoo(nn.Module):
    def __init__(self,input_size,hidden_size,output_size=1,num_layer=2):
        super(net_shampoo, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layer)
        self.linear = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        x,_ = self.lstm(x)
        s,b,h = x.shape
        x = x.view(s*b,h)
        x = self.linear(x)
        x = x.view(s,b,-1)
        return x

net = net_shampoo(input_size=2,hidden_size=10)
loss_func = nn.MSELoss()
lr = 0.01
optim = torch.optim.Adam(net.parameters(),lr=lr)
global_step = 0
epochs = 500

#训练模型
net.train()
for i in range(epochs):
    print('--------第{}轮训练开始----------'.format(global_step+1))
    output = net(trainX)
    loss = loss_func(output,trainY)
    optim.zero_grad()
    loss.backward()
    optim.step()
    global_step += 1
    print('第{}轮训练的损失：{}'.format(global_step,loss.item()))

#保存模型
torch.save(net,'finish_net')
print('模型已保存')

#预测数据
net.eval()
pred_test = net(testX)

#反归一化
pred_test = pred_test.view(-1).data.numpy()*(max - min) + min
testY = testY.view(-1).data.numpy()*(max - min) + min

#绘制图像
plt.plot(testY,label='real')
plt.plot(pred_test,label='prediction')
plt.legend(loc='best')
plt.show()



# 定义评估函数
def evaluate_model(testY, pred_test):
    mse = np.mean((testY - pred_test) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(testY - pred_test))

    print(f'MSE: {mse:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'MAE: {mae:.4f}')

    # 计算 R² 分数
    ss_res = np.sum((testY - pred_test) ** 2)
    ss_tot = np.sum((testY - np.mean(testY)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    print(f'R²: {r2:.4f}')

evaluate_model(testY, pred_test)










