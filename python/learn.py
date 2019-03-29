""" learn module """

from decimal import Decimal as D
from time import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from storage import load, save

processedJAFFE = load('processedJAFFE')
data = [element[1] for element in processedJAFFE]
labels = [element[2] for element in processedJAFFE]
processedJAFFE = None
# svc = OneVsRestClassifier(SVC(random_state=0, decision_function_shape='ovr',
#                               C=0.002, kernel='rbf',  gamma=0.0000001))
# scores = cross_val_score(svc, data, labels, cv=5, verbose=1, n_jobs=4)

# svc = OneVsRestClassifier(SVC(random_state=0, decision_function_shape='ovr',
#                               C=200, kernel='rbf',  gamma=0.000000001), n_jobs=3)
# Set the parameters by cross-validation
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, random_state=42)

parameters = {
    'C': [
        2e-8, 4e-8, 7e-8,
        2e-7, 4e-7, 7e-7,
        2e-6, 4e-6, 7e-6,
        2e-5, 4e-5, 7e-5,
        2e-4, 4e-4, 7e-4,
        2e-3, 4e-3, 7e-3,
    ],
    'gamma': [
        0.001, 0.0005, 0.0002, 0.00015
    ],
    # 'gamma': [1e-3, 1e-4, 2e-4, 3e-4, ],
    # 'C': [1e-3, 1e-4]
}
models = []
a = time()
for gamma in parameters['gamma']:
    for C in parameters['C']:
        print('C = %s \t gamma = %s'%(C, gamma))
        svc = OneVsRestClassifier(SVC(random_state=0, decision_function_shape='ovr',
                                      C=C, kernel='rbf',  gamma=gamma), n_jobs=1)
        # print(cross_val_score(svc, data, labels, scoring='accuracy', cv=5, n_jobs=3))
        svc.fit(X_train, y_train)
        y_true, y_pred = y_test, svc.predict(X_test)
        # print(classification_report(y_true, y_pred))
        y_true = np.array(y_true, dtype=np.unicode_)
        y_pred = np.array(y_pred, dtype=np.unicode_)
        correct = np.sum(y_true == y_pred)

        print("accuracy: %d/%d = "%(correct, len(y_true)), D('%.2f'%(correct/len(y_true)*100)))
        models.append([C, gamma, D('%.2f'%(correct/len(y_true)*100))])
        print('--------------------------')
print(models)
print('time: ', D('%.2f'%(time() - a)), "sec")
# print("saving model as 'model'")
save('values', models)
