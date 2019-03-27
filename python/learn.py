from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from storage import load, save
from time import time
import numpy as np
from decimal import Decimal as D

processedJAFFE = load('processedJAFFE')
data = [element[1] for element in processedJAFFE]
labels = [element[2] for element in processedJAFFE]
# svc = OneVsRestClassifier(SVC(random_state=0, decision_function_shape='ovr',
#                               C=0.002, kernel='rbf',  gamma=0.0000001))
# scores = cross_val_score(svc, data, labels, cv=5, verbose=1, n_jobs=4)

svc = OneVsRestClassifier(SVC(random_state=0, decision_function_shape='ovr',
                              C=0.002, kernel='rbf',  gamma=0.0000001), n_jobs=3)
# Set the parameters by cross-validation
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, random_state=42)

tuned_parameters = [{
    'estimator__kernel': ['rbf'],
    'estimator__gamma': [1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
    'estimator__C': [0.002, 0.0002, 0.0004, 0.00001, 0.000001]
}]

param_grid = {
    'kernel': ['rbf'],
    'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10],
    'C': [0.001, 0.0001, 0.1],
}
a = time()
svc.fit(X_train, y_train)
y_true, y_pred = y_test, svc.predict(X_test)
print(classification_report(y_true, y_pred))
y_true = np.array(y_true, dtype=np.unicode_)
y_pred = np.array(y_pred, dtype=np.unicode_)
correct = np.sum(y_true == y_pred)

print("accuracy: %d/%d = "%(correct, len(y_true)), D('%.2f'%(correct/len(y_true)*100)))
print('time: ', D('%.2f'%(time() - a)), "sec")
print("saving model as 'model'")
save('model', svc)
