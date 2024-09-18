#引入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch


#读取数据
def pre_process(data_name):
    data_csv = pd.read_csv(data_name)
    data = data_csv['Sales']

    '''plt.figure(figsize=(10,6))
    plt.plot(data)
    plt.show()
    '''
    #数据预处理
    dataset = data.values
    max_value = np.max(dataset)
    min_value = np.min(dataset)
    scalar = max_value - min_value
    dataset = list(map(lambda x : (x - min_value) / max_value, dataset))

    #创建数据集
    step = 2
    dataX, dataY = [], []
    for i in range(len(dataset) - step):
        a = dataset[i:i+step]
        b = dataset[i+step]
        dataX.append(a)
        dataY.append(b)
    dataX = np.array(dataX)
    dataY = np.array(dataY)

    train_size = int(len(dataX) * 0.6)
    trainX = dataX[:train_size]
    trainY = dataY[:train_size]
    testX = dataX[train_size:]
    testY = dataY[train_size:]

    trainX = trainX.reshape(-1,1,2)
    trainY = trainY.reshape(-1,1,1)
    testX = testX.reshape(-1,1,2)
    testY = testY.reshape(-1,1,1)

    trainX = torch.from_numpy(trainX).float()
    trainY = torch.from_numpy(trainY).float()
    testX = torch.from_numpy(testX).float()
    testY = torch.from_numpy(testY).float()

    return trainX, trainY, testX, testY, max_value, min_value

