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


    X_train, X_valid, y_train, y_valid = train_test_split(X_train_processed.drop(['PassengerId'],axis=1), Y_train_processed, test_size=0.2,
                                                          random_state=np.random.seed())


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


    G = pd.DataFrame({
        'PassengerId': test_dataset['PassengerId'],
        'Survived': g
    })


    G.to_csv('./submission/gbm_cabinX1.csv', index=False)

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
