""" classes accuracies module """

import argparse
from decimal import Decimal as D
from pathlib import Path
import csv
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from storage import load

# pylint: disable=invalid-name
def generateClassesAccuracies(filePath):
    """ generate csv containing accuracy per class
    """
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
    cgamma = [
        (2e-3, 1e-2),
        (1e-2, 1e-3),
        (1e-1, 1.1e-4),
        (1e-1, 7e-5),
        (1e+1, 1e-6),
        (1e+2, 1e-7),
    ]
    output = []
    output.append(['C', 'Gamma', 'AN', 'DI', 'FE', 'HA', 'NE', 'SA', 'SU'])
    for C, gamma in cgamma:
        row = [C, gamma]
        print('C = %s \t gamma = %s'%(C, gamma))
        svc = OneVsRestClassifier(SVC(random_state=0, decision_function_shape='ovr',
                                      C=C, kernel='rbf', gamma=gamma), n_jobs=4)
        svc.fit(xTrain, yTrain)
        yTrue, yPred = yTest, svc.predict(xTest)
        yTrue = np.array(yTrue, dtype=np.unicode_)
        yPred = np.array(yPred, dtype=np.unicode_)
        correct = np.sum(yTrue == yPred)
        
        print("accuracy: %d/%d = "%(correct, len(yTrue)),
              D('%.2f'%(correct/len(yTrue)*100)))
        report = classification_report(yTrue, yPred, output_dict=True)
        row.append(report['AN']['precision'])
        row.append(report['DI']['precision'])
        row.append(report['FE']['precision'])
        row.append(report['HA']['precision'])
        row.append(report['NE']['precision'])
        row.append(report['SA']['precision'])
        row.append(report['SU']['precision'])
        
        output.append(row)
    
    with open('../csv/%s_precision.csv'%filename, 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(output)

parser = argparse.ArgumentParser()
parser.add_argument('file')
filepath = Path(parser.parse_args().file)

generateClassesAccuracies(filepath)
