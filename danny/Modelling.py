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

from danny.preprocessing_2 import dataset

def main():
    train_dataset = pd.read_csv("./data/train.csv")
    test_dataset = pd.read_csv("./data/test.csv")

    X_train_processed = dataset[:891]
    Y_train_processed = train_dataset['Survived']
    test_processed = dataset[891:]


    X_train, X_valid, y_train, y_valid = train_test_split(X_train_processed, Y_train_processed, test_size=0.2,
                                                          random_state=np.random.seed())


    # ---------- GradientBoostingClassifier -------------
    GBM = GradientBoostingClassifier(learning_rate=0.1, n_estimators=50, max_depth=5)
    GBM.fit(X_train, y_train)
    print("Train score: {0.2f}", GBM.score(X_train, y_train))
    print("Valid score: {0.2f}", GBM.score(X_valid, y_valid))

    gbm_pred = GBM.predict(X_valid)
    print("Valid Accuracy is ", accuracy_score(y_valid, gbm_pred) * 100)
    g = GBM.predict(test_processed)

    # ----------------------- catboost ----------------
    # cat = CatBoostClassifier()
    # cat.fit(X_train, y_train)
    # print("Train score: {0.2f}", cat.score(X_train, y_train))
    # print("Valid score: {0.2f}", cat.score(X_valid, y_valid))
    #
    # cat_pred = cat.predict(X_valid)
    # # print("Valid Accuracy is ", accuracy_score(y_valid, cat_pred) * 100)
    # c = cat.predict(test_processed)


    # ----------------------- xgboost ----------------
    # xg = XGBClassifier(max_depth=5, learning_rate=0.4, gamma=3, min_child_weight=1, booster='dart')
    # xg.fit(X_train, y_train)
    # print("Train score: {0.2f}", xg.score(X_train, y_train))
    # print("Valid score: {0.2f}", xg.score(X_valid, y_valid))
    #
    # xg_pred = xg.predict(X_valid)
    # print("Valid Accuracy is ", accuracy_score(y_valid, xg_pred) * 100)
    # x = xg.predict(test_processed)



    G = pd.DataFrame({
        'PassengerId': test_dataset['PassengerId'],
        'Survived': g
    })

    # C = pd.DataFrame({
    #     'PassengerId': test_dataset['PassengerId'],
    #     'Survived': c
    # })

    # X = pd.DataFrame({
    #     'PassengerId': test_dataset['PassengerId'],
    #     'Survived': x
    # })



    G.to_csv('./submission/modelling.csv', index=False)
    # C.to_csv('cat_testing.csv', index=False)
    # X.to_csv('xgboost_testing.csv', index=False)

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
