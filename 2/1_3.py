import pandas as pn
import numpy as np


data_set = pn.read_csv(open(r'D:\Admin\Desktop\python_labs\2\housing.csv'))

#1)
length = len(data_set)
train_dt, validate_dt, test_dt = np.split(data_set, [int(.7*length), int(.85*length)])

#2)
sign = 'ocean_proximity'
for i in data_set[sign].unique():
    data_set[sign + '=' + i] = (data_set[sign] == i).astype(int)

#3)
data_set[['total_rooms', 'total_bedrooms']] = data_set[['total_rooms','total_bedrooms']].div(data_set['households'], axis = 0)
data_set.rename(columns={'total_rooms': 'average_rooms', 'total_bedrooms': 'average_bedrooms'}, inplace = True)

#4)
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

#5)
data_set['longitude'] = data_set['longitude'] / data_set['longitude'].abs().max()
data_set['latitude'] = data_set['latitude'] / data_set['latitude'].abs().max()


data_set.to_csv(r'D:\Admin\Desktop\python_labs\2\output.csv', index = False)