import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('data/jehun4.csv')
test = pd.read_csv('data/jehun4_test.csv')
rtrain = pd.read_csv('data/train.csv')
rtest = pd.read_csv('data/test.csv')
label = rtrain['Survived']

X_train, X_vali, y_train, y_vali = train_test_split(train, label, test_size=0.45)

n_iters = 0
kfold = KFold(n_splits = 3)

for train_idx, vali_idx in kfold.split(train):
    n_iters += 1
    print('iteration {} ==============='.format(n_iters))
    X_ktrain, X_kvali = train.iloc[train_idx, :], train.iloc[vali_idx, :]
    y_ktrain, y_kvali = label[train_idx], label[vali_idx]

    # LGBMClassifier + GridSearchCV
    if False:
        lgbm = LGBMClassifier()
        lgbm_param_grid = {

        }
        gs_lgbm = GridSearchCV(lgbm, param_grid=lgbm_param_grid, cv=kfold)
        gs_lgbm.fit(X_ktrain, y_ktrain)
        best = gs_lgbm.best_estimator_
        predicted = best.predict(X_kvali)
        accuracy = accuracy_score(y_kvali, predicted)
        print('(LGBM) accuracy : {0:0.2f}%'.format(accuracy * 100))
    
    # GradientBoostingClassifier + GridSearchCV
    if False:
        pass
    
    # VotingClassifier
    if True:
        # Logistic Regression
        lr = LogisticRegression(random_state=np.random.seed())
        lr.fit(X_ktrain, y_ktrain)
        print('(LR) accuracy : {0:0.2f}%'.format(lr.score(X_kvali, y_kvali) * 100))

        # SVC + GridSearchCV
        svc = SVC()
        svc_param_grid = {
            'kernel':['rbf'],
            'gamma':[0.01,0.1,1],
            'C':[1,10,50,100]
        }
        gs_svc = GridSearchCV(svc,param_grid=svc_param_grid,cv=kfold, scoring="accuracy", n_jobs=4)
        gs_svc.fit(X_ktrain, y_ktrain)
        svc_best = gs_svc.best_estimator_
        #FIXME
        svc_score = gs_svc.best_score_
        print('(SVC_gs) accuracy : {0:0.2f}%'.format(svc_score * 100))

        # Random Forest Classifier + GridSearchCV
        rfc = RandomForestClassifier(random_state=np.random.seed())
        rfc_param_grid = {
            "max_depth":[None],
            "max_features":['auto','sqrt','log2'],
            "min_samples_split":[3,10],
            "min_samples_leaf":[1,3,10],
            "bootstrap":[False],
            "n_estimators":[200,700],
            "criterion":["gini"]
        }
        gs_rfc = GridSearchCV(rfc,param_grid=rfc_param_grid, cv=kfold, scoring="accuracy", n_jobs=4)
        gs_rfc.fit(X_ktrain, y_ktrain)
        rfc_best = gs_rfc.best_estimator_
        #FIXME
        rfc_score = gs_rfc.best_score_
        print('(RFC_gs) accuracy : {0:0.2f}%'.format(rfc_score * 100))

        # Gradient Boosting Classifier
        gbc = GradientBoostingClassifier(loss='deviance', learning_rate=0.05, n_estimators=100)
        gbc.fit(X_ktrain, y_ktrain)
        print("(GBC) train accuracy : {0:0.2f}%".format(gbc.score(X_ktrain,y_ktrain) * 100))
        print("(GBC) accuracy : {0:0.2f}%".format(gbc.score(X_kvali,y_kvali) * 100))

        # Voting Classifier
        vc = VotingClassifier(
            estimators=[('lr', lr), ('rfc', rfc_best), ('svc', svc_best), ('gbc',gbc)],
            voting='hard', n_jobs=4)
        # vc = VotingClassifier(
        #     estimators=[('rfc', rfc_best), ('svc', svc_best), ('gbc',gbc)],
        #     voting='hard', n_jobs=4)
        vc.fit(X_ktrain, y_ktrain)
        print("(VC) accuracy : {0:0.2f}%".format(vc.score(X_kvali, y_kvali) * 100))

        predicted_vc = vc.predict(test)
        predicted_vc = pd.DataFrame({
        'PassengerId': rtest['PassengerId'],
        'Survived' : predicted_vc
        })
        predicted_vc.to_csv('data/vc{}.csv'.format(n_iters), index=False)
