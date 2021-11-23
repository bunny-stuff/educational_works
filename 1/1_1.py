import json
import csv


sales_js = json.load(open('sales.json'))
sales_csv = csv.writer(open('sales.csv', 'w', newline='', encoding='utf-8'))

sales_csv.writerow(["item", "country", "years", "sales"])

for sale in sales_js:
    print(sale["item"])
    for country in sale['sales_by_country']:
        print(country)
        for year in sale['sales_by_country'][country]:
            print(year)
            print(sale['sales_by_country'][country][year])
            sales_csv.writerow([sale["item"], country, year, sale['sales_by_country'][country][year]])