import urllib
import csv

name = 'tory'
c = 1

with open('members.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        f = open(name+str(c)+'.jpg','wb')
        f.write(urllib.urlopen(row[2]).read())
        f.close()
        c += 1


