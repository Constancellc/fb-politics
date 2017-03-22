import numpy as np
import random
import csv

parties = ['green','labour','lib-dem','tory','ukip']
labels = {'green':0,'labour':1,'lib-dem':2,'tory':3,'ukip':4}

def getData():
    X = []
    y = []

    Xt = []
    yt = []
    
    for party in parties:
        with open('rgb_averages/'+str(party)+'_results.csv','rU') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                ran = random.random()
                if ran >= 0.2:
                    X.append(row)
                    y.append(labels[party])
                else:
                    Xt.append(row)
                    yt.append(labels[party])
                    
    return [X, y, Xt, yt]
