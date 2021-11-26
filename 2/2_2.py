import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from matplotlib import pyplot as plt


data_set = pd.read_excel(r'D:\Admin\Desktop\python_labs\2\ccpp.xlsx', sheet_name='Sheet1')
print(data_set.shape)

length = len(data_set['AT'])
train_set, test_set = np.split(data_set, [int(.7*length)])

x_tensor = torch.tensor(train_set[['AT', 'V', 'AP', 'RH']].values).float()
y_tensor = torch.tensor(train_set['PE'].values).float()
#print(x_tensor.shape)
#print(y_tensor.shape)

#x_tensor = torch.transpose(x_tensor, 0, -1)
y_tensor = y_tensor.unsqueeze(1)
#print(x_tensor.shape)
#print(y_tensor.shape)


class LinearRegression(nn.Module):
    def __init__(self, inputSize, outputSize, neurons):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(inputSize, outputSize)

    def forward(self, X):
        predictions = self.linear(X) #<-- return nan
        return predictions

model = LinearRegression(4, 1, 1000)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01)

epochs = 2000
for epoch in range(epochs):
    optimizer.zero_grad()
    predictions = model(x_tensor)
    #print(predictions)
    loss = criterion(predictions, y_tensor)
   
    loss.backward()
   
    optimizer.step()
    
    if epoch % 100 == 0:
        print('epoch {}, loss {}'.format(epoch, loss.item()))
