import csv
from storage import load

data = load('kirsch_hvnLBP_JAFFE1')
with open('kirsch_hvnLBP_JAFFE1.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(data)