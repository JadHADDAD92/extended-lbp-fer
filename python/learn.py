""" learn module """

from decimal import Decimal as D
from time import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from storage import load, save

processedJAFFE = load('../data/kirsch_LBP_JAFFE')
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
        5.00E-04,
        2.00E-04,
        1.50E-04,
        1.10E-04,
        1.05E-04,
        1.00E-04,
        9.50E-05,
        9.00E-05,
        7.00E-05,
        5.00E-05,
        1.90E-05,
        1.00E-05,
        1.00E-06,
        1.00E-07,
    ],
}

models = []
models.append(['gamma \ C', 1.00E-12, 1.00E-11, 1.00E-10, 1.00E-09, 1.00E-08,
               1.00E-07, 1.00E-06, 1.00E-05, 1.00E-04, 1.00E-03, 2.00E-03, 1.00E-02,
               1.00E-01, 1.00, 1.00E+01, 1.00E+02, 1.00E+03, 1.00E+04, 1.00E+05 ])
a = time()
for gamma in parameters['gamma']:
    row = [gamma]
    for C in parameters['C']:
        b = time()
        print('C = %s \t gamma = %s'%(C, gamma))
        svc = OneVsRestClassifier(SVC(random_state=0, decision_function_shape='ovr',
                                      C=C, kernel='rbf',  gamma=gamma), n_jobs=1)
        svc.fit(xTrain, yTrain)
        yTrue, yPred = yTest, svc.predict(xTest)
        yTrue = np.array(yTrue, dtype=np.unicode_)
        yPred = np.array(yPred, dtype=np.unicode_)
        correct = np.sum(yTrue == yPred)
        
        print("accuracy: %d/%d = "%(correct, len(yTrue)),
              D('%.2f'%(correct/len(yTrue)*100)))
        row.append(D('%.2f'%(correct/len(yTrue)*100)))
        print('--------------------------(%d sec)'%(time()-b))
    models.append(row)
print(models)
save('kirsch_LBP_JAFFE', models)
print('time: ', D('%.2f'%(time() - a)), "sec")

# ------------------------------------------------------------------------------

processedJAFFE = load('../data/kirsch_LBP_JAFFE1')
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

models = []
models.append(['gamma \ C', 1.00E-12, 1.00E-11, 1.00E-10, 1.00E-09, 1.00E-08,
               1.00E-07, 1.00E-06, 1.00E-05, 1.00E-04, 1.00E-03, 2.00E-03, 1.00E-02,
               1.00E-01, 1.00, 1.00E+01, 1.00E+02, 1.00E+03, 1.00E+04, 1.00E+05 ])
a = time()
for gamma in parameters['gamma']:
    row = [gamma]
    for C in parameters['C']:
        b = time()
        print('C = %s \t gamma = %s'%(C, gamma))
        svc = OneVsRestClassifier(SVC(random_state=0, decision_function_shape='ovr',
                                      C=C, kernel='rbf',  gamma=gamma), n_jobs=1)
        svc.fit(xTrain, yTrain)
        yTrue, yPred = yTest, svc.predict(xTest)
        yTrue = np.array(yTrue, dtype=np.unicode_)
        yPred = np.array(yPred, dtype=np.unicode_)
        correct = np.sum(yTrue == yPred)
        
        print("accuracy: %d/%d = "%(correct, len(yTrue)),
              D('%.2f'%(correct/len(yTrue)*100)))
        row.append(D('%.2f'%(correct/len(yTrue)*100)))
        print('--------------------------(%d sec)'%(time()-b))
    models.append(row)
print(models)
save('kirsch_LBP_JAFFE1', models)
print('time: ', D('%.2f'%(time() - a)), "sec")

# ------------------------------------------------------------------------------

processedJAFFE = load('../data/kirsch_hvnLBP_JAFFE')
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

models = []
models.append(['gamma \ C', 1.00E-12, 1.00E-11, 1.00E-10, 1.00E-09, 1.00E-08,
               1.00E-07, 1.00E-06, 1.00E-05, 1.00E-04, 1.00E-03, 2.00E-03, 1.00E-02,
               1.00E-01, 1.00, 1.00E+01, 1.00E+02, 1.00E+03, 1.00E+04, 1.00E+05 ])
a = time()
for gamma in parameters['gamma']:
    row = [gamma]
    for C in parameters['C']:
        b = time()
        print('C = %s \t gamma = %s'%(C, gamma))
        svc = OneVsRestClassifier(SVC(random_state=0, decision_function_shape='ovr',
                                      C=C, kernel='rbf',  gamma=gamma), n_jobs=1)
        svc.fit(xTrain, yTrain)
        yTrue, yPred = yTest, svc.predict(xTest)
        yTrue = np.array(yTrue, dtype=np.unicode_)
        yPred = np.array(yPred, dtype=np.unicode_)
        correct = np.sum(yTrue == yPred)
        
        print("accuracy: %d/%d = "%(correct, len(yTrue)),
              D('%.2f'%(correct/len(yTrue)*100)))
        row.append(D('%.2f'%(correct/len(yTrue)*100)))
        print('--------------------------(%d sec)'%(time()-b))
    models.append(row)
print(models)
save('kirsch_hvnLBP_JAFFE', models)
print('time: ', D('%.2f'%(time() - a)), "sec")

# ------------------------------------------------------------------------------

processedJAFFE = load('../data/kirsch_hvnLBP_JAFFE1')
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

models = []
models.append(['gamma \ C', 1.00E-12, 1.00E-11, 1.00E-10, 1.00E-09, 1.00E-08,
               1.00E-07, 1.00E-06, 1.00E-05, 1.00E-04, 1.00E-03, 2.00E-03, 1.00E-02,
               1.00E-01, 1.00, 1.00E+01, 1.00E+02, 1.00E+03, 1.00E+04, 1.00E+05 ])
a = time()
for gamma in parameters['gamma']:
    row = [gamma]
    for C in parameters['C']:
        b = time()
        print('C = %s \t gamma = %s'%(C, gamma))
        svc = OneVsRestClassifier(SVC(random_state=0, decision_function_shape='ovr',
                                      C=C, kernel='rbf',  gamma=gamma), n_jobs=1)
        svc.fit(xTrain, yTrain)
        yTrue, yPred = yTest, svc.predict(xTest)
        yTrue = np.array(yTrue, dtype=np.unicode_)
        yPred = np.array(yPred, dtype=np.unicode_)
        correct = np.sum(yTrue == yPred)
        
        print("accuracy: %d/%d = "%(correct, len(yTrue)),
              D('%.2f'%(correct/len(yTrue)*100)))
        row.append(D('%.2f'%(correct/len(yTrue)*100)))
        print('--------------------------(%d sec)'%(time()-b))
    models.append(row)
print(models)
save('kirsch_hvnLBP_JAFFE1', models)
print('time: ', D('%.2f'%(time() - a)), "sec")
