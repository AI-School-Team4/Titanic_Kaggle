import numpy as np
import pandas as pd

# headers
# PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked

def load_train_data():
    return pd.read_csv('data/train.csv')

def load_test_data():
    return pd.read_csv('data/test.csv')

def fill_na_with_median(train, test):
    train['Age'].fillna(train.groupby('Title')['Age'].transform('median'), inplace=True)
    test['Age'].fillna(test.groupby('Title')['Age'].transform('median'), inplace=True)
    
    train['Fare'].fillna(train.groupby('Pclass')['Fare'].transform('median'),inplace=True)
    test['Fare'].fillna(test.groupby('Pclass')['Fare'].transform('median'),inplace=True)

    train['Cabin'].fillna(train.groupby('Pclass')['Cabin'].transform('median'),inplace=True)
    test['Cabin'].fillna(test.groupby('Pclass')['Cabin'].transform('median'),inplace=True)

    train['FamilySize'] = train['SibSp'] + train['Parch']+1
    test['FamilySize'] = test['SibSp'] + test['Parch']+1

    return train, test


def delete_na_row(train, test):
    fdrop = ['Ticket', 'SibSp', 'Parch']
    train = train.drop(fdrop,axis=1)
    test = test.drop(fdrop,axis=1)
    #train = train.drop(['PassengerId'],axis=1)
    
    return train, test

def mapping(dataset, train, test):
    title_mapping = {'Mr':0,'Miss':1,'Mrs':2,'Master':3,'Rev':3,'Dr':3,'Col':3,'Major':3,'Mlle':3,'Countess':3,'Ms':3,'Lady':3,'Jonkheer':3,'Don':3,'Dona':3,'Mme':3,'Capt':3,'Sir':3}
    sex_mapping = {'male':0,'female':1}
    embarked_mapping = {'S':0,'C':1,'Q':2}
    cabin_mapping = {'A':0,'B':0.4,'C':0.8,'D':1.2,'E':1.6,'G':2.4,'T':2.8}
    
    for data in dataset:
        data['Title'] = data['Name'].str.extract('([A-Za-z]+)\.', expand=False)
        data['Embarked'] = data['Embarked'].fillna('S')
        data['Cabin'] = data['Cabin'].str[:1]
        
    for data in dataset:
        data['Title'] = data['Title'].map(title_mapping)
        data['Sex'] = data['Sex'].map(sex_mapping)
        data['Embarked'] = data['Embarked'].map(embarked_mapping)
        data['Cabin'] = data['Cabin'].map(cabin_mapping)
        
    
    train.drop('Name',axis=1,inplace=True)
    test.drop('Name',axis=1,inplace=True)
    return dataset, train, test

def mapping2(dataset):
    family_mapping = {1:0,2:0.4,3:0.8,4:1.2,5:1.6,6:2,7:2.4,8:2.8,9:3.2,10:3.6,11:4}
    for data in dataset:
        data.loc[data['Fare']<=17,'Fare']=0,
        data.loc[(data['Fare']>17) & (data['Fare']<=30),'Fare']=1,
        data.loc[(data['Fare']>30) & (data['Fare']<=100),'Fare']=2,
        data.loc[data['Fare']>100,'Fare']=3,
        data.loc[data['Age']<=16,'Age']=0,
        data.loc[(data['Age']>16)&(data['Age']<=26),'Age']=1,
        data.loc[(data['Age']>26)&(data['Age']<=36),'Age']=2,
        data.loc[(data['Age']>36)&(data['Age']<=62),'Age']=3,
        data.loc[data['Age']>62, 'Age']=4

        data['FamilySize'] = data['FamilySize'].map(family_mapping)
    return dataset

def preprocessing():
    train = load_train_data()
    test = load_test_data()
    
    train_test = [train,test]
    train_test, train, test = mapping(train_test, train, test)
    train, test = fill_na_with_median(train, test)
    train_test = mapping2(train_test)
    train, test = delete_na_row(train, test)
    
    return train, test
    
if __name__ == "__main__":
    preprocessing()
