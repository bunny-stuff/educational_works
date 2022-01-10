import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


data_set = pd.read_csv(r'D:\Admin\Desktop\python_labs\2\Davis.csv')
data_set.replace('', np.nan, inplace=True)
data_set.dropna(inplace=True)

train_val_set, test_set = train_test_split(data_set, test_size = 50)
train_set, val_set = train_test_split(train_val_set, test_size = 0.3)

data_set = data_set.loc[data_set['height'] > 140]

set_m = train_set.query('sex == "M"')
set_f = train_set.query('sex == "F"')

fig, ax = plt.subplots(2, 3, figsize=(12, 6))
ax[0][0].set_title('Height all')
ax[0][0].set_xlim(145, 195)
ax[0][0].hist(train_set['height'], bins = 12)

ax[0][1].set_title('Height M')
ax[0][1].set_xlim(145, 195)
ax[0][1].hist(set_m['height'], bins = 12)

ax[0][2].set_title('Height F')
ax[0][2].set_xlim(145, 195)
ax[0][2].hist(set_f['height'], bins = 12)


ax[1][0].set_title('Weight all')
ax[1][0].set_xlim(35, 120)
ax[1][0].hist(data_set['weight'], bins = 12)

ax[1][1].set_title('Weight M')
ax[1][1].set_xlim(35, 120)
ax[1][1].hist(set_m['weight'], bins = 12)

ax[1][2].set_title('Weight F')
ax[1][2].set_xlim(35, 120)
ax[1][2].hist(set_f['weight'], bins = 12)

X_train, y_train = train_set[['height', 'weight']], train_set['sex']

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

predictions = logreg.predict(X_train)
accuracy = accuracy_score(predictions, y_train)
print(f'Accuracy: {accuracy:.2f}')

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle(f'Accuracy: {accuracy:.2f}')
centres_prediction = logreg.predict(list(itertools.product(np.linspace(150, 190, 5), np.linspace(45, 95, 5))))

def convert_sex(value):
       value = value.replace('F','1').replace('M', '0')
       return int(value)

centres_prediction = np.vectorize(convert_sex)(np.array(centres_prediction)).reshape(5, 5)


x = [40, 52, 64, 76, 88, 100]
y = [145, 155, 165, 175, 185, 195]

cmap = ListedColormap(['blue', 'orange'])
ax[0].pcolormesh(x, y, centres_prediction, alpha=0.4, cmap=cmap)
ax[1].pcolormesh(x, y, centres_prediction, alpha=0.4, cmap=cmap)

ax[0].plot(X_train, y_train)

ax[0].set_title('Train')
ax[0].scatter(set_m['weight'], set_m['height'])
ax[0].scatter(set_f['weight'], set_f['height'], color='orange')
ax[0].set_ylim(145, 195)
ax[0].set_xlim(40, 100)

test_m = test_set.query('sex == "M"')
test_f = test_set.query('sex == "F"')

ax[1].set_title('Test')
ax[1].scatter(test_m['weight'], test_m['height'])
ax[1].scatter(test_f['weight'], test_f['height'], color='orange')
ax[1].set_ylim(145, 195)
ax[1].set_xlim(40, 100)

plt.show()