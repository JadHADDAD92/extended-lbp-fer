""" learn module """

from decimal import Decimal as D
from time import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from storage import load

processedJAFFE = load('processedJAFFE')
data = [element[1] for element in processedJAFFE]
labels = [element[2] for element in processedJAFFE]
processedJAFFE = None
xTrain, xTest, yTrain, yTest = train_test_split(data,
                                                labels,
                                                test_size=0.33,
                                                random_state=42)

parameters = {
    'C': [
        1e-5, 1e-4, 1e-3
    ],
    'gamma': [
        1.9e-5
    ],
}

models = []
a = time()
for gamma in parameters['gamma']:
    for C in parameters['C']:
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
        models.append([C, gamma, D('%.2f'%(correct/len(yTrue)*100))])
        print('--------------------------')
print(models)
print('time: ', D('%.2f'%(time() - a)), "sec")
