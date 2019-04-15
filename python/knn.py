""" learn module """

from decimal import Decimal as D
from time import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from storage import load, save

processedJAFFE = load('../data/dLBP45_JAFFE(2)')
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


neigh = KNeighborsClassifier(n_neighbors=1, p=1)
neigh.fit(xTrain, yTrain)
yTrue, yPred = yTest, neigh.predict(xTest)
yTrue = np.array(yTrue, dtype=np.unicode_)
yPred = np.array(yPred, dtype=np.unicode_)
correct = np.sum(yTrue == yPred)

print("accuracy: %d/%d = "%(correct, len(yTrue)),
        D('%.2f'%(correct/len(yTrue)*100)))
