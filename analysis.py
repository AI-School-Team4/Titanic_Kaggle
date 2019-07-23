import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import export_graphviz, DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

from preprocessing import preprocessor


def main():

    train_dataset = pd.read_csv("./data/train.csv")
    test_dataset = pd.read_csv("./data/test.csv")

    # pre-processing
    X_train_processed, Y_train_processed, test_processed = preprocessor(train_dataset, test_dataset, fill_age_with='advanced_median_1', fill_cabin_with='X',
                                          dropPassengerID=False, dropName=True, dropTicket=True)
    # print('xtrain-------------------------\n')
    # print(X_train.head())
    # print('ytrain---------------------------\n')
    # print(Y_train.head())
    # print('test---------------------------\n')
    # print(test)
    # print(test.columns)

    X_train, X_valid, y_train, y_valid = train_test_split(X_train_processed.drop(['PassengerId'],axis=1), Y_train_processed, test_size=0.2,
                                                          random_state=np.random.seed())

    #########
    # train #
    #########

    # kfold = KFold(n_splits=10, shuffle=True, random_state=np.random.seed())
    # svc = SVC(probability=True)
    # svc_param_grid = {
    #     'kernel': ['rbf'],
    #     'gamma': [0.01, 0.1, 1],
    #     'C': [1, 10, 50, 100]
    # }
    # gs_svc = GridSearchCV(svc, param_grid=svc_param_grid, cv=kfold, scoring="accuracy", n_jobs=4)
    # gs_svc.fit(X_train, y_train)
    # svc_best = gs_svc.best_estimator_
    # svc_score = gs_svc.best_score_
    # print('svc_score:', svc_score)
    #
    # rfc = RandomForestClassifier()
    # param_grid = {
    #     "max_depth": [3, 4, 5, 6, 7, 8, 9, 10, 11],
    #     "max_features": ['auto', 'sqrt', 'log2'],
    #     "min_samples_split": [3, 10],
    #     "min_samples_leaf": [1, 3, 10],
    #     "bootstrap": [False],
    #     "n_estimators": [200, 700],
    #     "criterion": ["gini"]
    # }
    # gs_rfc = GridSearchCV(rfc, param_grid=param_grid, cv=kfold, scoring="accuracy", n_jobs=4)
    # gs_rfc.fit(X_train, y_train)
    # rfc_best = gs_rfc.best_estimator_
    # rfc_score = gs_rfc.best_score_
    # print('rfc_score:', rfc_score)
    #
    # log_clf = LogisticRegression(random_state=42)
    # nvc = GaussianNB()
    #
    # # rnd_clf = RandomForestClassifier(random_state=42)
    # # svm_clf = SVC(random_state=42)
    # voting_clf = VotingClassifier(
    #     estimators=[('lr', log_clf), ('rf', rfc_best), ('svc', svc_best), ('nvc', nvc)],
    #     voting='soft', n_jobs=4)
    # voting_clf.fit(X_train, y_train)
    #
    # for clf in (log_clf, rfc_best, svc_best, voting_clf):
    #     clf.fit(X_train, y_train)
    #     y_pred = clf.predict(X_valid)
    #     print(clf.__class__.__name__, accuracy_score(y_valid, y_pred))
    # prediction = voting_clf.predict(test_processed.drop('PassengerId', axis=1))
    #
    # # train(KFold + SVC)
    #
    # k_fold = KFold(n_splits=10, shuffle=True, random_state=np.random.seed())
    # clf = KNeighborsClassifier(n_neighbors=13)
    # scoring = 'accuracy'
    # score = cross_val_score(clf, X_train_processed, Y_train_processed, cv=k_fold, n_jobs=4, scoring=scoring)
    # print(score)
    # print(round(np.mean(score) * 100, 2))
    #
    # clf = SVC()
    # score = cross_val_score(clf, X_train_processed, Y_train_processed, cv=k_fold, n_jobs=4, scoring=scoring)
    # print(score)
    # print(round(np.mean(score) * 100, 2))
    #
    # clf.fit(X_train, y_train)
    # prediction = clf.predict(test_processed.drop('PassengerId', axis=1))
    #
    # tree_clf = DecisionTreeClassifier(max_depth=4, random_state=np.random.seed())
    # tree_clf.fit(X_train, y_train)
    # print("Train score: {0.2f}", tree_clf.score(X_train, y_train))
    # print("Valid score: {0.2f}", tree_clf.score(X_valid, y_valid))
    #
    # valid_pred = tree_clf.predict(X_valid)
    # print("Valid Accuracy is ", accuracy_score(y_valid, valid_pred) * 100)
    # p = tree_clf.predict(test_processed.drop('PassengerId', axis=1))


    # ---------- GradientBoostingClassifier -------------
    GBM = GradientBoostingClassifier(loss='deviance', learning_rate=0.05, n_estimators=100)
    GBM.fit(X_train, y_train)
    print("Train score: {0.2f}", GBM.score(X_train, y_train))
    print("Valid score: {0.2f}", GBM.score(X_valid, y_valid))

    gbm_pred = GBM.predict(X_valid)
    print("Valid Accuracy is ", accuracy_score(y_valid, gbm_pred) * 100)
    g = GBM.predict(test_processed.drop('PassengerId', axis=1))

    # --------- Lightgbm -----------
    # lgb = lightgbm()
    # lgb.fit(X_train, y_train)
    # print("Train score: {0.2f}", lgb.score(X_train, y_train))
    # print("Valid score: {0.2f}", lgb.score(X_valid, y_valid))
    #
    # lgb_pred = lgb.predict(X_valid)
    # print("Valid Accuracy is ", accuracy_score(y_valid, lgb_pred) * 100)
    # l = lgb.predict(test_processed.drop('PassengerId', axis=1))




    # VC = pd.DataFrame({
    #     'PassengerId': test['PassengerId'],
    #     'Survived': prediction
    # })
    #
    # DT = pd.DataFrame({
    #     'PassengerId': test['PassengerId'],
    #     'Survived': p
    # })

    G = pd.DataFrame({
        'PassengerId': test_dataset['PassengerId'],
        'Survived': g
    })

    # L = pd.DataFrame({
    #     'PassengerId': test_processed['PassengerId'],
    #     'Survived': l
    # })

    # VC.to_csv('vc.csv', index=False)
    # DT.to_csv('dt.csv', index=False)

    # G.to_csv('./submission/gbm_cabinX1.csv', index=False)
    # L.to_csv('./submission/lgb.csv', index=False)

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
