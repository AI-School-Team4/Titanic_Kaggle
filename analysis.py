import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from preprocessing import preprocessor


# Gaussian Naive Bayes
def get_nvc(X_train, Y_train):
    nvc = GaussianNB()
    nvc.fit(X_train, Y_train)
    score = nvc.score(X_train, Y_train)
    return nvc, score


# SVM classifier grid search
def get_svc(X_train, Y_train, kfold):
    svc = SVC(probability=True)

    svc_param_grid = {
        'kernel':['rbf'],
        'gamma' : [0.01, 0.1, 1],
        'C' : [1, 10, 50, 100]
    }

    gs_svc = GridSearchCV(svc, param_grid=svc_param_grid, cv=kfold, scoring="accuracy", n_jobs=4)
    gs_svc.fit(X_train, Y_train)

    svc_best = gs_svc.best_estimator_

    score = gs_svc.best_score_

    return svc_best, score


# Random Forest Classifier Grid Search
def get_rfc(X_train, Y_train, kfold):
    rfc = RandomForestClassifier()

    rf_param_grid = {
        "max_depth" : [None],
        "max_features" : [3, 10],
        "min_samples_split" : [3, 10],
        "min_samples_leaf" : [1, 3, 10],
        "bootstrap" : [False],
        "n_estimators" : [100, 300],
        "criterion" : ["gini"]
    }

    gs_rfc = GridSearchCV(rfc, param_grid=rf_param_grid, cv=kfold, scoring="accuracy", n_jobs=4)

    gs_rfc.fit(X_train, Y_train)

    rfc_best = gs_rfc.best_estimator_

    score = gs_rfc.best_score_

    return rfc_best, score


# nvc, svc, rfc -> Ensemble Classifier
def ensemble_voting(X_train, Y_train, nvc, svc, rfc):

    votingC = VotingClassifier(estimators=[('nvc', nvc), ('svc', svc), ('rfc', rfc)],
                               voting='soft', n_jobs=4)

    votingC = votingC.fit(X_train, Y_train)

    score = votingC.score(X_train, Y_train)

    return votingC, score


def main():

    train_dataset = pd.read_csv("./data/train.csv")
    test_dataset = pd.read_csv("./data/test.csv")

    # pre-processing
    X_train, Y_train, test = preprocessor(train_dataset, test_dataset, fill_age_with='median', fill_cabin_with='mapping_median',
                                          dropPassengerID=True, dropName=False)
    print('xtrain-------------------------\n')
    print(X_train.head())
    print('ytrain---------------------------\n')
    print(Y_train.head())
    print('test---------------------------\n')
    print(test)
    print(test.columns)

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
