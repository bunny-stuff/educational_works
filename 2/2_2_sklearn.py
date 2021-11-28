import pandas as pd
from sklearn import linear_model
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score


data_set = pd.read_excel(r'D:\Admin\Desktop\python_labs\2\ccpp.xlsx', sheet_name='Sheet1')

length = len(data_set['AT'])
valid_set, test_set = np.split(data_set, [int(.7*length)])

x_valid = valid_set[['AT', 'V', 'AP', 'RH']].to_numpy().reshape(-1,4)
y_valid = valid_set['PE']

x_test = test_set[['AT', 'V', 'AP', 'RH']].to_numpy().reshape(-1,4)
y_test = test_set['PE']

ols = linear_model.LinearRegression()
model = ols.fit(x_valid, y_valid)

test_predicts = model.predict(x_test)
plt.plot(y_test, test_predicts, 'bo', markersize = 1)
xpoints = ypoints = plt.xlim()
plt.plot(xpoints, ypoints, color = 'k', lw = 1.5)

valid_predicts = model.predict(x_valid)
r2 = r2_score(y_valid, valid_predicts)
av_err = (y_valid - valid_predicts).mean()
plt.title(f'R^2 = {r2: .5f}\n Средняя ошибка = {av_err}')

plt.show()