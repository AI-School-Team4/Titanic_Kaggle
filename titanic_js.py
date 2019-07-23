import pandas as pd
import numpy as np
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
#import pydot
import warnings
warnings.filterwarnings("ignore")

# dataset 불러오기
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

# feature preprocessing
train_test_data = [train, test]
for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]+)\.', expand=False)
    
title_mapping = {'Mr':0,'Miss':1,'Mrs':2,'Master':3,'Rev':3,'Dr':3,'Col':3,'Major':3,'Mlle':3,'Countess':3,'Ms':3,'Lady':3,'Jonkheer':3,'Don':3,'Dona':3,'Mme':3,'Capt':3,'Sir':3}
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)
train.drop('Name',axis=1,inplace=True)
test.drop('Name',axis=1,inplace=True)


sex_mapping = {'male':0,'female':1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)

train['Age'].fillna(train.groupby('Title')['Age'].transform('median'), inplace=True)
test['Age'].fillna(test.groupby('Title')['Age'].transform('median'), inplace=True)

for dataset in train_test_data:
    dataset.loc[dataset['Age']<=16,'Age']=0,
    dataset.loc[(dataset['Age']>16)&(dataset['Age']<=26),'Age']=1,
    dataset.loc[(dataset['Age']>26)&(dataset['Age']<=36),'Age']=2,
    dataset.loc[(dataset['Age']>36)&(dataset['Age']<=62),'Age']=3,
    dataset.loc[dataset['Age']>62, 'Age']=4

Pclass1 = train[train['Pclass']==1]['Embarked'].value_counts()
Pclass2 = train[train['Pclass']==2]['Embarked'].value_counts()
Pclass3 = train[train['Pclass']==3]['Embarked'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])

for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

embarked_mapping = {'S':0,'C':1,'Q':2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)

train['Fare'].fillna(train.groupby('Pclass')['Fare'].transform('median'),inplace=True)
test['Fare'].fillna(test.groupby('Pclass')['Fare'].transform('median'),inplace=True)

for dataset in train_test_data:
    dataset.loc[dataset['Fare']<=17,'Fare']=0,
    dataset.loc[(dataset['Fare']>17) & (dataset['Fare']<=30),'Fare']=1,
    dataset.loc[(dataset['Fare']>30) & (dataset['Fare']<=100),'Fare']=2,
    dataset.loc[dataset['Fare']>100,'Fare']=3

for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]

cabin_mapping = {'A':0,'B':0.4,'C':0.8,'D':1.2,'E':1.6,'G':2.4,'T':2.8}
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)

train['Cabin'].fillna(train.groupby('Pclass')['Cabin'].transform('median'),inplace=True)
test['Cabin'].fillna(test.groupby('Pclass')['Cabin'].transform('median'),inplace=True)

train['FamilySize'] = train['SibSp'] + train['Parch']+1
test['FamilySize'] = test['SibSp'] + test['Parch']+1

family_mapping = {1:0,2:0.4,3:0.8,4:1.2,5:1.6,6:2,7:2.4,8:2.8,9:3.2,10:3.6,11:4}
for dataset in train_test_data:
    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)

feature_drop = ['Ticket', 'SibSp', 'Parch']
train = train.drop(feature_drop, axis=1)
test = test.drop(feature_drop, axis=1)

train = train.drop(['PassengerId'],axis=1)
train_data = train.drop('Survived', axis=1)
target = train['Survived']
#print(train_data.shape, target.shape)

#print(train_data.info())

# train (VotingClassifier)
X_train, X_valid, y_train, y_valid = train_test_split(train_data, target, test_size=0.2, random_state = np.random.seed())


###################
####           ####
####   train   ####
####           ####
###################  

kfold = KFold(n_splits=10,shuffle=True,random_state=np.random.seed())
svc = SVC(probability=True)
svc_param_grid = {
    'kernel':['rbf'],
    'gamma':[0.01,0.1,1],
    'C':[1,10,50,100]
}
gs_svc = GridSearchCV(svc,param_grid=svc_param_grid,cv=kfold, scoring="accuracy", n_jobs=4)
gs_svc.fit(X_train, y_train)
svc_best = gs_svc.best_estimator_
svc_score = gs_svc.best_score_
print('svc_score:',svc_score)

rfc = RandomForestClassifier()
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
gs_rfc.fit(X_train, y_train)
rfc_best = gs_rfc.best_estimator_
rfc_score = gs_rfc.best_score_
print('rfc_score:',rfc_score)

log_clf = LogisticRegression(random_state=42)
nvc = GaussianNB()

# rnd_clf = RandomForestClassifier(random_state=42)
# svm_clf = SVC(random_state=42)
voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rfc_best), ('svc', svc_best), ('nvc',nvc)],
    voting='soft', n_jobs=4)
voting_clf.fit(X_train, y_train)

for clf in (log_clf, rfc_best, svc_best, voting_clf):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_valid)
    print(clf.__class__.__name__, accuracy_score(y_valid,y_pred))
test_data = test.drop('PassengerId', axis=1).copy()
prediction = voting_clf.predict(test_data)



# train(KFold + SVC)

k_fold = KFold(n_splits=10,shuffle=True,random_state=np.random.seed())
clf = KNeighborsClassifier(n_neighbors=13)
scoring = 'accuracy'
score = cross_val_score(clf,train_data,target,cv=k_fold,n_jobs=4,scoring=scoring)
print(score)
print(round(np.mean(score)*100,2))

clf = SVC()
score = cross_val_score(clf,train_data,target,cv=k_fold,n_jobs=4,scoring=scoring)
print(score)
print(round(np.mean(score)*100,2))

clf.fit(train_data, target)
test_data = test.drop('PassengerId',axis=1).copy()
prediction = clf.predict(test_data)

tree_clf = DecisionTreeClassifier(max_depth=4, random_state=np.random.seed())
tree_clf.fit(X_train,y_train)
print("Train score: {0.2f}", tree_clf.score(X_train,y_train))
print("Valid score: {0.2f}", tree_clf.score(X_valid,y_valid))

valid_pred = tree_clf.predict(X_valid)
print("Valid Accuracy is ", accuracy_score(y_valid, valid_pred)*100)
p = tree_clf.predict(test_data)

# export_graphviz(tree_clf, out_file = "decisionTree1.dot",class_names=["malignant","benign"],impurity=False)

# (graph, )=pydot.graph_from_dot_file('decisionTree1.dot',encoding='utf8')

# graph.write_png('decisionTree1.png')

GBM = GradientBoostingClassifier(loss='deviance', learning_rate=0.05, n_estimators=100)
GBM.fit(X_train, y_train)
print("Train score: {0.2f}", GBM.score(X_train,y_train))
print("Valid score: {0.2f}", GBM.score(X_valid,y_valid))

valid_pred = GBM.predict(X_valid)
print("Valid Accuracy is ", accuracy_score(y_valid, valid_pred)*100)
g = GBM.predict(test_data)


VC = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived' : prediction
})

DT = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived' : p
})

G = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': g
})

VC.to_csv('vc.csv',index=False)
DT.to_csv('dt.csv',index=False)
G.to_csv('gbm.csv',index=False)
