""" meta learner module
"""
from decimal import Decimal as D
import numpy as np
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from storage import load

def traintestsplit(filepath):
    """ split data into train/test subsets
    """
    proba1 = load(filepath)
    processedDF = pd.DataFrame(proba1)
    
    # sort emotion dictionary and get its values
    processedDF[1] = processedDF[1].apply(lambda y: sorted(y.items(), key=lambda x: x[0]))
    processedDF[1] = processedDF[1].apply(lambda y: list(map(lambda x: x[1], y)))
    
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

filesPaths = [
    '../data/probabilities/dLBP90_JAFFE_probabilities_c_100.0_gamma_0.01',
    '../data/probabilities/TLP1_14x14_JAFFE_probabilities_c_1000.0_gamma_7e-05',
    '../data/probabilities/TLP2_14x14_JAFFE_probabilities_c_100.0_gamma_0.0005',
    '../data/probabilities/TLP2_16x16_JAFFE_probabilities_c_1000.0_gamma_0.0001',
    '../data/probabilities/dLBP45_14x14_JAFFE_probabilities_c_1000.0_gamma_0.0002',
]

traintests = np.array([traintestsplit(filePath) for filePath in filesPaths])
xTrains, yTrains, xTests, yTests = traintests.transpose()

def concatResults(arrays):
    """ concatenate the arrays
    """
    return list(np.hstack(elems) for elems in zip(*arrays))

def mulitplyResults(arrays):
    """ multiply elements wise the arrays
    """
    return list(np.prod(elems, axis=0) for elems in zip(*arrays))

# pylint: disable=no-member
def maxResults(arrays):
    """ pick the max of the arrays
    """
    return list(np.maximum.reduce(elems) for elems in zip(*arrays))

def meanResults(arrays):
    """ compute the average of the arrays
    """
    multiplier = 1/len(arrays)
    return list(multiplier * np.sum(elems, axis=0) for elems in zip(*arrays))

parameters = {
    'C': [ float(base*D(10)**power)
           for power in range(-12, 11, 1)
           for base in range(1, 10) ],
    'gamma': [ float(1*D(10)**power) for power in range(-12, 11, 1)],
}

yTrain = yTrains[0]
yTest = yTests[0]
xTrain = concatResults(xTrains)
xTest = concatResults(xTests)

maxScore = 0
bestC = []
bestGamma = []
for gamma in parameters['gamma']:
    for C in parameters['C']:
        svc = OneVsRestClassifier(SVC(random_state=0, decision_function_shape='ovr',
                                      C=C, kernel='rbf', gamma=gamma),
                                  n_jobs=4)
        svc.fit(xTrain, yTrain)
        yTrue, yPred = yTest, svc.predict(xTest)
        yTrue = np.array(yTrue, dtype=np.unicode_)
        yPred = np.array(yPred, dtype=np.unicode_)
        correct = np.sum(yTrue == yPred)
        print('C=%s\tgamma=%s'%(C, gamma))
        accuracy = correct/len(yTrue)*100
        print("accuracy: %d/%d = "%(correct, len(yTrue)), D('%.2f'%(accuracy)))
        if accuracy >= maxScore:
            maxScore = accuracy
            bestC.append(C)
            bestGamma.append(gamma)
print('maxScore', maxScore)
print('C, Gamma', *zip(bestC, bestGamma))
