import requests
import xml.etree.ElementTree as xmlTree
from datetime import datetime
import matplotlib.pyplot as ppl
 

date_formatter = "%d.%m.%Y"
today = datetime.now().strftime('%d/%m/%Y')

def get_ready_array(currency_code):
	res = requests.get(f"http://www.cbr.ru/scripts/XML_dynamic.asp?date_req1=01/01/2021&date_req2={today}&VAL_NM_RQ={currency_code}")
	tree = xmlTree.fromstring(res._content)
	date_zero = datetime.strptime('01.01.2021', '%d.%m.%Y')
	ar = []
	for element in tree:
		date_string = element.get('Date')
		date = datetime.strptime(date_string, date_formatter)
		days_from_zero = (date - date_zero).days
		#print(date, days_from_zero)
		value = float(element.find('Value').text.replace(',','.'))
		ar.append([days_from_zero, value])
	return ar

def get_x_axis(matrix):
	x = []
	for i in matrix:
		x.append(i[0])
	return x

def get_y_axis(matrix):
	y = []
	for i in matrix:
		y.append(i[1])
	return y


dollars = get_ready_array('R01235')
dollars_x = get_x_axis(dollars)
dollars_y = get_y_axis(dollars)

euros = get_ready_array('R01239')
euros_x = get_x_axis(euros)
euros_y = get_y_axis(euros)

yapanse_yen = get_ready_array('R01820')
yapanse_yen_x = get_x_axis(yapanse_yen)
yapanse_yen_y = get_y_axis(yapanse_yen)

hryvnia = get_ready_array('R01720')
hryvnia_x = get_x_axis(hryvnia)
hryvnia_y = get_y_axis(hryvnia)

fgr, ax = ppl.subplots(2, 2, figsize=(10, 8))
fgr.suptitle('Курсы валют к рублю', fontsize = 30)

ax[0, 0].set_title('Доллар')
ax[0, 0].plot(dollars_x, dollars_y)

ax[0, 1].set_title('Евро')
ax[0, 1].plot(euros_x, euros_y)

ax[1, 0].set_title('Японская йена')
ax[1, 0].plot(yapanse_yen_x, yapanse_yen_y)

ax[1, 1].set_title('Украинская гривна')
ax[1, 1].plot(hryvnia_x, hryvnia_y)

ppl.show()
