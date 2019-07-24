import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForeastClassifier
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV

#data load
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X_train = train.drop(['Survived'], axis=1)
Y_train = train['Survived']

X_test = test

#print(X_train.head(5))
#print(X_test.head(5))

#data feature engineering
def pre_data(x):
    
    #embarked
    em_map = {'S':1, 'C':2, 'Q':3}
    x = x.fillna({'Embarked':'S'}) #fill the null-embarked
    x['Embarked'] = x['Embarked'].map(em_map)
    #print(x['Embarked'])
    
    #name
    title_map = {'Mr':1, 'Rare':2, 'Master':3, 'Miss':4, 'Mrs':5, 'Royal':6} 
    x['Title'] = x['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
    x['Title'] = x['Title'].replace(['Lady', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    x['Title'] = x['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
    x['Title'] = x['Title'].replace(['Mlle', 'Ms'], 'Miss')
    x['Title'] = x['Title'].replace('Mme', 'Mrs')
    x['Title'] = x['Title'].map(title_map)
    x['Title'] = x['Title'].fillna(0)
    #print(x['Title'])
    
    #sex
    sex_map = {'female':1, 'male':0}
    x['Sex'] = x['Sex'].map(sex_map)
    #print(x['Sex'])
    
    #age
    age_title_map = {1:'Young Adult', 2:'Adult', 3:'Baby', 4:'Student', 5:'Adult', 6:'Adult'}
    age_map = {'Baby':1, 'Child':2, 'Teenager':3, 'Student':4, 'Young Adult':5, 'Adult':6, 'Senior':7}
    x['Age'] = x['Age'].fillna(-0.5)
    bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
    labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    x['AgeGroup'] = pd.cut(x['Age'], bins, labels = labels)

    for z in range(len(x['AgeGroup'])):
        if x['AgeGroup'][z] == 'Unknown':
            x['AgeGroup'][z] = age_title_map[x['Title'][z]]
   
    x['AgeGroup'] = x['AgeGroup'].map(age_map)
    #print(x['AgeGroup'])
    
    #fare
    x['FareBend'] = pd.qcut(x['Fare'], 4, labels=[1,2,3,4])
    #print(x['Fare'])

    x = x.drop(['Name', 'Ticket', 'Fare', 'Cabin', 'Age'], axis=1)
    
    return x
    
X_train = pre_data(X_train)
X_test = pre_data(X_test)

X_train.info()
X_test.info()

#analysis
n_splits = 10
random_state = 0
n_estimators = 13
n_jobs = 1
scoring = 'accuracy'
param = {
    'n_estimators':[2,3],
    'max_depth':[2,3],
    'min_samples_leaf':[2,3],
    'min_samples_split':[2,3,4]
}

rfc = RandomForestClassifier(n_estimators = n_estimators)
grid_rfc = GridSearchCV(rfc, param_grid=param)
grid_rfc.fit(X_train, Y_train)
estimator = grid_rfc.best_estimator_

pred = estimator.predict(X_test)

k_fold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
score = cross_val_score(estimator, X_train, Y_train, cv=k_fold, n_jobs=n_jobs, scoring=scoring)

round(np.mean(score)*100, 2)
    
#save result
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': pred
})
submission.to_csv('submission.csv', index=False)
#print(submission.head(5))
