""" learn module """

import argparse
from decimal import Decimal as D
from datetime import timedelta
from pathlib import Path
from time import time
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from storage import load

def learn(filePath):
    filename = filePath.stem
    processedJAFFE = load(str(filePath))
    processedDF = pd.DataFrame(processedJAFFE)
    processedDF.columns = ['name', 'data', 'emotion']
    processedDF = processedDF.sort_values(by=['name', 'emotion'])
    grouped = processedDF.groupby(['name', 'emotion'])
    train = grouped.nth([0, 1])
    test = grouped.nth([2, 3, 4])

    yTrain = train.index.get_level_values(1).tolist()
    xTrain = train.values.ravel().tolist()
    yTest = test.index.get_level_values(1).tolist()
    xTest = test.values.ravel().tolist()

    parameters = {
        'C': [
            1.00E-12, 1.00E-11, 1.00E-10, 1.00E-09, 1.00E-08, 1.00E-07, 1.00E-06,
            1.00E-05, 1.00E-04, 1.00E-03, 2.00E-03, 1.00E-02, 1.00E-01, 1.00,
            1.00E+01, 1.00E+02, 1.00E+03, 1.00E+04, 1.00E+05
        ],
        'gamma': [
            1.00E00,
            1.00E-01,
            1.00E-02,
            1.00E-03,
            5.00E-04, 2.00E-04, 1.50E-04, 1.10E-04, 1.05E-04, 1.00E-04,
            9.50E-05, 9.00E-05, 7.00E-05, 5.00E-05, 1.90E-05, 1.00E-05,
            1.00E-06,
            1.00E-07,
        ],
    }

    models = []
    models.append(['gamma \ C', 1.00E-12, 1.00E-11, 1.00E-10, 1.00E-09, 1.00E-08,
                   1.00E-07, 1.00E-06, 1.00E-05, 1.00E-04, 1.00E-03, 2.00E-03, 1.00E-02,
                   1.00E-01, 1.00, 1.00E+01, 1.00E+02, 1.00E+03, 1.00E+04, 1.00E+05 ])
    gridTimeStart = time()
    numIteration = len(parameters['gamma']) * len(parameters['C'])
    iteration = 0
    meanTime = 0
    for gamma in parameters['gamma']:
        row = [gamma]
        for C in parameters['C']:
            print('C = %s \t gamma = %s'%(C, gamma))
            timeStart = time()
            svc = OneVsRestClassifier(SVC(random_state=0, decision_function_shape='ovr',
                                        C=C, kernel='rbf', gamma=gamma), n_jobs=4)
            svc.fit(xTrain, yTrain)
            yTrue, yPred = yTest, svc.predict(xTest)
            yTrue = np.array(yTrue, dtype=np.unicode_)
            yPred = np.array(yPred, dtype=np.unicode_)
            correct = np.sum(yTrue == yPred)
            
            print("accuracy: %d/%d = "%(correct, len(yTrue)),
                D('%.2f'%(correct/len(yTrue)*100)))
            row.append(D('%.2f'%(correct/len(yTrue)*100)))
            
            iterTime = time()-timeStart
            iteration = iteration + 1
            meanTime = meanTime * (iteration-1)/iteration + iterTime/iteration
            remainingTime = (numIteration-iteration)*meanTime
            print('--------------------------(%d sec)--remaining: %s'%
                  (iterTime, str(timedelta(seconds=int(remainingTime)))))
        models.append(row)
    gridTime = time() - gridTimeStart
    gridTime = timedelta(seconds=int(gridTime))
    print('time: %s'%str(gridTime))
    print('saving file: %s.csv'%filename)
    with open('%s.csv'%filename, 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(models)

parser = argparse.ArgumentParser()
parser.add_argument('file')
filepath = Path(parser.parse_args().file)

learn(filepath)
