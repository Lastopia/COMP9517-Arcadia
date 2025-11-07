from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def train_svm(X, y, tune=False):

    if tune:
        param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 10, 1, 0.01, 0.001]}
        grid = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=3)
        grid.fit(X, y)
        print("Best params:", grid.best_params_)
        return grid.best_estimator_
    else:
        clf = svm.SVC(kernel='rbf', C=10, gamma='scale')
        clf.fit(X, y)
        return clf

def evaluate_svm(clf, X_test):
    preds = clf.predict(X_test)
    return preds
    # print("\nConfusion Matrix:\n", confusion_matrix(y_test, preds))
    # print("\nClassification Report:\n", classification_report(y_test, preds))
