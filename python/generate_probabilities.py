""" compute probabilities
"""
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from storage import load, save

# pylint: disable=invalid-name
def extractProbabilities(filePath, C, gamma):
    """ build dataset containing the probabilities of each class of every image
        in the test set from the original dataset
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
    
    svc = OneVsRestClassifier(SVC(random_state=0, decision_function_shape='ovr',
                                  C=C, kernel='rbf', gamma=gamma, probability=True),
                              n_jobs=4)
    svc.fit(xTrain, yTrain)
    classes = svc.classes_
    probas = svc.predict_proba(xTest)
    dataset = []
    for index, proba in enumerate(probas):
        dictProba = dict(zip(classes, proba))
        dataset.append([dictProba, yTest[index]])
    newFilename = filename+'_probabilities_testdata_c_%s_gamma_%s'%(C, gamma)
    print('saving file:', '/data/%s'%newFilename)
    save('../data/probabilities/%s'%newFilename, dataset)

parser = argparse.ArgumentParser()
parser.add_argument('file')
parser.add_argument('c', type=float)
parser.add_argument('gamma', type=float)

args = parser.parse_args()
extractProbabilities(Path(args.file), args.c, args.gamma)
