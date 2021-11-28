import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score


data_set = pd.read_excel(r'D:\Admin\Desktop\python_labs\2\ccpp.xlsx', sheet_name='Sheet1')
print(data_set.shape)
#нормализация
#data_set['AT'] = data_set['AT'] / data_set['AT'].abs().max()
#data_set['V'] = data_set['V'] / data_set['V'].abs().max()
#data_set['AP'] = data_set['AP'] / data_set['AP'].abs().max()
#data_set['RH'] = data_set['RH'] / data_set['RH'].abs().max()
#data_set['PE'] = data_set['PE'] / data_set['PE'].abs().max()

length = len(data_set['AT'])
valid_set, test_set = np.split(data_set, [int(.7*length)])

x_tensor = torch.tensor(valid_set[['AT', 'V', 'AP', 'RH']].values).float()
y_tensor = torch.tensor(valid_set['PE'].values).float()
y_tensor = y_tensor.unsqueeze(1)


class LinearRegression(nn.Module):
    def __init__(self, inputSize, outputSize, neurons):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(inputSize, outputSize)

    def forward(self, x):
        predictions = self.linear(x)
        return predictions

model = LinearRegression(4, 1, 1000)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)

epochs = 6000
for epoch in range(epochs):
    optimizer.zero_grad()
    predictions = model(x_tensor)
    loss = criterion(predictions, y_tensor)
   
    loss.backward()
   
    optimizer.step()
    
    if epoch % 500 == 0:
        print('epoch {}, loss {}'.format(epoch, loss.item()))


x_test_tensor = torch.tensor(test_set[['AT', 'V', 'AP', 'RH']].values).float()
y_test_tensor = torch.tensor(test_set['PE'].values).float()
y_test_tensor = y_test_tensor.unsqueeze(1)

plt.plot(y_test_tensor.detach().numpy(), model(x_test_tensor).detach().numpy(), 'bo', markersize = 1)
xpoints = ypoints = plt.xlim()
plt.plot(xpoints, ypoints, color = 'k', lw = 1.5)

valid_preds = model.forward(x_tensor)
r2 = r2_score(y_tensor.detach().numpy(), valid_preds.detach().numpy())
av_err = (y_tensor.detach().numpy() - valid_preds.detach().numpy()).mean()
plt.title(f'R^2 = {r2: .5f}\n Средняя ошибка = {av_err}')

plt.show()