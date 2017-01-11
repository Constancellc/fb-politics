import csv
import numpy as np

c = 1

index = range(0,49,2)

while True:
    if c == 247:
        c+= 1
    with open('data/tory'+str(c)+'.csv') as csvfile:
        reader = csv.reader(csvfile)

        array = []

        for row in reader:
            array.append(row)

    downsampled = []

    for j in range(0,25):
        for i in index:
            r = (float(array[100*j+i][0])+float(array[100*j+i+1][0])+
                 float(array[100*j+50][0])+float(array[100*j+51][0]))/4
            g = (float(array[100*j+i][1])+float(array[100*j+i+1][1])+
                 float(array[100*j+50][1])+float(array[100*j+51][1]))/4
            b = (float(array[100*j+i][2])+float(array[100*j+i+1][2])+
                 float(array[100*j+50][2])+float(array[100*j+51][2]))/4
            downsampled.append([r,g,b])

    with open('downsampled/tory'+str(c)+'.csv','w') as csvfile:
        writer = csv.writer(csvfile)

        for row in downsampled:
            writer.writerow(row)

    c += 1

    
