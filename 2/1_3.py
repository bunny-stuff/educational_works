import pandas as pn
import numpy as np
from sklearn import preprocessing


data_set = pn.read_csv(open('housing.csv'))

#2) Dummy
sign = 'ocean_proximity'
for i in data_set[sign].unique():
    data_set[sign + '=' + i] = (data_set[sign] == i).astype(int)

#3) Меняем на средние значения
data_set[['total_rooms', 'total_bedrooms']] = data_set[['total_rooms','total_bedrooms']].div(data_set['households'], axis = 0)
data_set.rename(columns={'total_rooms': 'average_rooms', 'total_bedrooms': 'average_bedrooms'}, inplace = True)

#1) Делим датасет на выборки
length = len(data_set)
train_dt, validate_dt, test_dt = np.split(data_set, [int(.8*length), int(.9*length)])
#print(f"train: {train_dt.shape}")
#print(f"validate: {validate_dt.shape}")
#print(f"test: {test_dt.shape}")

#5) Нормализация
scaler_longitude = preprocessing.StandardScaler().fit(train_dt[['longitude']])
scaler_latitude = preprocessing.StandardScaler().fit(train_dt[['latitude']])

train_dt['longitude'] = scaler_longitude.transform(train_dt[['longitude']]).reshape(-1)
train_dt['latitude'] = scaler_latitude.transform(train_dt[['latitude']]).reshape(-1)

validate_dt['longitude'] = scaler_longitude.transform(validate_dt[['longitude']]).reshape(-1)
validate_dt['latitude'] = scaler_latitude.transform(validate_dt[['latitude']]).reshape(-1)

test_dt['longitude'] = scaler_longitude.transform(test_dt[['longitude']]).reshape(-1)
test_dt['latitude'] = scaler_latitude.transform(test_dt[['latitude']]).reshape(-1)

#print(train_dt['longitude'].to_numpy().mean(axis = 0))
#print(train_dt['longitude'].to_numpy().std(axis = 0))

#print(test_dt['longitude'].to_numpy().mean(axis = 0))
#print(test_dt['longitude'].to_numpy().std(axis = 0))

#4) 
print(train_dt.isna().sum())
print(validate_dt.isna().sum())
print(test_dt.isna().sum())
print("--")
train_average_rooms_mean = train_dt['average_bedrooms'].dropna().mean()
print(train_dt.isna().sum())
print("--")
train_dt['average_bedrooms'].fillna(value=train_average_rooms_mean, inplace = True)
validate_dt['average_bedrooms'].fillna(value=train_average_rooms_mean, inplace = True)
test_dt['average_bedrooms'].fillna(value=train_average_rooms_mean, inplace = True)

print(train_dt.isna().sum())
print(validate_dt.isna().sum())
print(test_dt.isna().sum())

exit

count_null_average_bedrooms = data_set['average_bedrooms'].isnull().sum()
#пустые значения заполняем средним количеством комнат в одной местности, 
#так мы не завысим и не занизим среднее количество комнат на одной местности
def calculate_average_if_need(bedrooms, ocean_p):
    if (np.isnan(bedrooms)):
        filter = data_set[sign] == ocean_p
        average = data_set[filter]['average_bedrooms'].mean()
        return average
    else:
        return bedrooms

data_set['average_bedrooms'] = data_set.apply(lambda row: calculate_average_if_need(row['average_bedrooms'], row[sign]), axis = 1)


data_set.to_csv('output.csv', index = False)