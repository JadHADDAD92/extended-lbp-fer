""" compute probabilities
"""
import argparse
from pprint import pprint
from pathlib import Path
import pandas as pd
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
    
    # extract train data
    train = grouped.nth([0, 1])
    yTrain = train.index.get_level_values(1).tolist()
    xTrain = train.values.ravel().tolist()
    
    # train our model
    svc = OneVsRestClassifier(SVC(random_state=0, decision_function_shape='ovr',
                                  C=C, kernel='rbf', gamma=gamma, probability=True),
                              n_jobs=4)
    svc.fit(xTrain, yTrain)
    
    classes = svc.classes_
    for index, (_name, data, _emotion) in enumerate(processedJAFFE):
        proba = svc.predict_proba(data.reshape(1, -1))[0]
        processedJAFFE[index][1] = dict(zip(classes, proba))
        pprint(processedJAFFE[index])
        print(svc.predict(data.reshape(1, -1)))
        print('-'*50)
    newFilename = filename+'_probabilities_c_%s_gamma_%s'%(C, gamma)
    print('saving file:', '/data/%s'%newFilename)
    save('../data/probabilities/%s'%newFilename, processedJAFFE)

parser = argparse.ArgumentParser()
parser.add_argument('file')
parser.add_argument('c', type=float)
parser.add_argument('gamma', type=float)

args = parser.parse_args()
extractProbabilities(Path(args.file), args.c, args.gamma)
