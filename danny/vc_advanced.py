from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from danny.preprocessing_1 import preprocessor
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd


def main():
    train_dataset = pd.read_csv("../data/train.csv")
    test_dataset = pd.read_csv("../data/test.csv")

    # pre-processing
    X_train_processed, Y_train_processed, test_processed = preprocessor(train_dataset, test_dataset,
                                                                        fill_age_with='advanced_median_1',
                                                                        fill_cabin_with='X',
                                                                        dropPassengerID=False, dropName=True,
                                                                        dropTicket=True)

    X_train, X_valid, y_train, y_valid = train_test_split(X_train_processed.drop(['PassengerId'], axis=1),
                                                          Y_train_processed, test_size=0.2,
                                                          random_state=np.random.seed())



    # log_clf = LogisticRegression(random_state=42)
    rnd_clf = RandomForestClassifier(random_state=42)
    svm_clf = SVC(random_state=42)
    gbm_clf = GradientBoostingClassifier(random_state=42)
    # cat_clf = CatBoostClassifier(random_state=42)
    xg_clf = XGBClassifier(random_state=42)
    voting_clf = VotingClassifier(
        estimators=[('gbm', gbm_clf), ('rnd', rnd_clf), ('svm', svm_clf),('xg', xg_clf)],
        voting='hard')
    voting_clf.fit(X_train, y_train)
    print("Train score: {0.2f}", voting_clf.score(X_train, y_train))
    print("Valid score: {0.2f}", voting_clf.score(X_valid, y_valid))
    v = voting_clf.predict(test_processed.drop('PassengerId', axis=1))


    V = pd.DataFrame({
        'PassengerId': test_dataset['PassengerId'],
        'Survived': v
    })


    V.to_csv('../submission/vc_advanced.csv', index=False)

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
