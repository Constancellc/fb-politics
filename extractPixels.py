import csv

from PIL import Image

#n = len(photos)
sourceName = 'photos/tory'
destinationName = 'data/tory'

c = 1
while True:
    print c
    im = Image.open(sourceName+str(c)+'.jpg','r')
    pix = list(im.getdata())

    with open(destinationName+str(c)+'.csv','w') as csvfile:
        writer = csv.writer(csvfile)
        for row in pix:
            try:
                writer.writerow(row)
            except:
                # greyscale images
                writer.writerow([row,row,row])
    c += 1


