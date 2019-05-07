""" meta learner module
"""
from decimal import Decimal as D
import numpy as np
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from pprint import pprint
from storage import load

def traintestsplit(filepath):
    proba1 = load(filepath)
    processedDF = pd.DataFrame(proba1)
    processedDF[1] = processedDF[1].apply(lambda x: list(x.values()))
    processedDF.columns = ['name', 'data', 'emotion']
    processedDF = processedDF.sort_values(by=['name', 'emotion'])
    grouped = processedDF.groupby(['name', 'emotion'])

    # split train and test data
    train = grouped.nth([0, 1])
    test = grouped.nth([2, 3, 4])
    _yTrain = train.index.get_level_values(1).tolist()
    _xTrain = train.values.ravel().tolist()
    _yTest = test.index.get_level_values(1).tolist()
    _xTest = test.values.ravel().tolist()
    return _xTrain, _yTrain, _xTest, _yTest

xTrain1, yTrain1, xTest1, yTest1  = traintestsplit('../data/probabilities/dLBP90_JAFFE_probabilities_c_100.0_gamma_0.01')
xTrain2, yTrain2, xTest2, yTest2  = traintestsplit('../data/probabilities/TLP1_14x14_JAFFE_probabilities_c_1000.0_gamma_7e-05')
xTrain3, yTrain3, xTest3, yTest3  = traintestsplit('../data/probabilities/TLP2_14x14_JAFFE_probabilities_c_100.0_gamma_0.0005')
xTrain4, yTrain4, xTest4, yTest4  = traintestsplit('../data/probabilities/TLP2_16x16_JAFFE_probabilities_c_1000.0_gamma_0.0001')

# concatenate feature vectors
xTrainConcat = list(elem1+elem2+elem3+elem4 for (elem1, elem2, elem3, elem4) in zip(xTrain1, xTrain2, xTrain3, xTrain4))
xTestConcat  = list(elem1+elem2+elem3+elem4 for (elem1, elem2, elem3, elem4) in zip(xTest1, xTest2, xTest3, xTest4))

# multiply feature vectors
xTrainMultiply = list(np.array(elem1)*np.array(elem2)*np.array(elem3)*np.array(elem4) for (elem1, elem2, elem3, elem4) in zip(xTrain1, xTrain2, xTrain3, xTrain4))
xTestMultiply  = list(np.array(elem1)*np.array(elem2)*np.array(elem3)*np.array(elem4) for (elem1, elem2, elem3, elem4) in zip(xTest1, xTest2, xTest3, xTest4))

# max feature vectors
xTrainMax = list(np.maximum.reduce([np.array(elem1), np.array(elem2), np.array(elem3), np.array(elem4)])
                 for (elem1, elem2, elem3, elem4) in zip(xTrain1, xTrain2, xTrain3, xTrain4))
xTestMax  = list(np.maximum.reduce([np.array(elem1), np.array(elem2), np.array(elem3), np.array(elem4)])
                 for (elem1, elem2, elem3, elem4) in zip(xTest1, xTest2, xTest3, xTest4))

# mean feature vectors
xTrainMean = list(0.25*(np.array(elem1)+np.array(elem2)+np.array(elem3)+np.array(elem4)) for (elem1, elem2, elem3, elem4) in zip(xTrain1, xTrain2, xTrain3, xTrain4))
xTestMean  = list(0.25*(np.array(elem1)+np.array(elem2)+np.array(elem3)+np.array(elem4)) for (elem1, elem2, elem3, elem4) in zip(xTest1, xTest2, xTest3, xTest4))


xTrain = xTrainMax
xTest  = xTestMax

yTrain = yTrain1
yTest = yTest1

parameters = {
    'C': [ base*10**pow  for pow in range(-12, 11, 1) for base in range(1, 10) ],
    'gamma': [1*10**pow for pow in range(-12, 11, 1)],
}
maxScore = 0
for gamma in parameters['gamma']:
    for C in parameters['C']:
        svc = OneVsRestClassifier(SVC(random_state=0, decision_function_shape='ovr',
                                    C=C, kernel='rbf', gamma=gamma, probability=True),
                                n_jobs=4)
        svc.fit(xTrain, yTrain)
        yTrue, yPred = yTest, svc.predict(xTest)
        yTrue = np.array(yTrue, dtype=np.unicode_)
        yPred = np.array(yPred, dtype=np.unicode_)
        correct = np.sum(yTrue == yPred)
        print('C=%s\tgamma=%s'%(C, gamma))
        accuracy = correct/len(yTrue)*100
        print("accuracy: %d/%d = "%(correct, len(yTrue)),
              D('%.2f'%(accuracy)))
        maxScore = accuracy if accuracy > maxScore else maxScore
print('maxScore', maxScore)
