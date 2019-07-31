import pandas as pd
import numpy as np
from collections import Counter

# Check if one feature is preProcessed or not
def status(feature):
    print('Processing', feature, ': ok')


def detect_outliers(df, n, features):
    """
    :param
    Dataframe, n, features (columns)

    :return
    list of the indices corresponding to the observations containing more than n outliers
    """
    outlier_indices = []

    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col], 75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1

        # outlier step
        outlier_step = 1.5 * IQR

        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index

        # append the found outlier indices for col to the list of outlier indices
        outlier_indices.extend(outlier_list_col)

    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)

    print(multiple_outliers)
    return multiple_outliers



'''
dataset is a combined data of train and test
    after dropping outliers
train ==> dataset[:881]
test  ==> dataset[881:]
'''
def combining_data():
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')

    # Dropping Survivors from training set
    train.drop(['Survived'], 1, inplace=True)

    # Dropping Outliers
    outliers_indices = detect_outliers(train, 2, ["Age", "SibSp", "Parch", "Fare"])     # 10개
    train = train.drop(outliers_indices, axis=0).reset_index(drop=True)

    # remove PassengerID
    dataset = train.append(test)
    dataset.reset_index(inplace=True)
    dataset.drop(['index', 'PassengerId'], inplace=True, axis=1)

    return dataset


dataset = combining_data()

#########
# Title #
#########
# Extract Title from Names
Title_Dictionary = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir": "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess": "Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr": "Mr",
    "Mrs": "Mrs",
    "Miss": "Miss",
    "Master": "Master",
    "Lady": "Royalty"
}


def process_titles():
    # we extract the title from each name
    dataset['Title'] = dataset['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())

    # a map of more aggregated title
    # we map each title
    dataset['Title'] = dataset.Title.map(Title_Dictionary)
    status('Title')
    return dataset


dataset = process_titles()

########
# Age  #
########
grouped_train = dataset.iloc[:881].groupby(['Sex', 'Pclass', 'Title'])
grouped_median_train = grouped_train.median()
grouped_median_train = grouped_median_train.reset_index()[['Sex', 'Pclass', 'Title', 'Age']]

def fill_age(row):
    condition = (
        (grouped_median_train['Sex'] == row['Sex']) &
        (grouped_median_train['Title'] == row['Title']) &
        (grouped_median_train['Pclass'] == row['Pclass'])
    )
    return grouped_median_train[condition]['Age'].values[0]


def process_age():
    global dataset
    # a function that fills the missing values of the Age variable
    dataset['Age'] = dataset.apply(lambda row: fill_age(row) if np.isnan(row['Age']) else row['Age'], axis=1)
    status('age')
    return dataset


dataset = process_age()

########
# Name #
########
def process_names():
    global dataset
    # drop Name
    dataset.drop('Name', axis=1, inplace=True)

    # dummy 명목 시리즈
    titles_dummies = pd.get_dummies(dataset['Title'], prefix='Title')
    dataset = pd.concat([dataset, titles_dummies], axis=1)

    # removing the title variable
    dataset.drop('Title', axis=1, inplace=True)

    status('names')
    return dataset


dataset = process_names()

########
# Fare #
########
def process_fares():
    global dataset
    # Fill one missing fare data with mean
    dataset.Fare.fillna(dataset.iloc[:881].Fare.mean(), inplace=True)
    # Log scaled
    dataset["Fare"] = dataset["Fare"].map(lambda x: np.log(x) if x > 0 else 0)

    status('fare')
    return dataset


dataset = process_fares()

############
# Embarked #
############
def process_embarked():
    global dataset
    # S is the most common one
    dataset.Embarked.fillna('S', inplace=True)
    # dummy 명목변수
    embarked_dummies = pd.get_dummies(dataset['Embarked'], prefix='Embarked')
    dataset = pd.concat([dataset, embarked_dummies], axis=1)
    dataset.drop('Embarked', axis=1, inplace=True)
    status('embarked')
    return dataset


dataset = process_embarked()


#########
# Cabin #
#########
train_cabin, test_cabin = set(), set()

for c in dataset.iloc[:881]['Cabin']:
    try:
        train_cabin.add(c[0])
    except:
        train_cabin.add('U')

for c in dataset.iloc[881:]['Cabin']:
    try:
        test_cabin.add(c[0])
    except:
        test_cabin.add('U')

def process_cabin():
    global dataset
    # replacing missing cabins with U (for Uknown)
    dataset.Cabin.fillna('U', inplace=True)

    # mapping each Cabin value with the cabin letter
    dataset['Cabin'] = dataset['Cabin'].map(lambda c: c[0])

    # dummy 명목변수
    cabin_dummies = pd.get_dummies(dataset['Cabin'], prefix='Cabin')
    dataset = pd.concat([dataset, cabin_dummies], axis=1)

    dataset.drop('Cabin', axis=1, inplace=True)
    status('cabin')
    return dataset


dataset = process_cabin()

#######
# Sex #
#######
def process_sex():
    global dataset
    # mapping string values to numerical one
    dataset['Sex'] = dataset['Sex'].map({'male':1, 'female':0})
    status('Sex')
    return dataset


dataset = process_sex()


##################
# Pclass and Sex #
##################
def process_pclass_sex():
    global dataset
    '''
    <Processing High chances>
    1. Pclass 3 & male(1)
    2. Pclass 1 & female(0)
    '''
    dataset['Pc3_male'] = dataset.apply(lambda row: 10 if row.Pclass==3 and row.Sex==1 else 0, axis=1)
    dataset['Pc1_female'] = dataset.apply(lambda row: 10 if row.Pclass==1 and row.Sex==0 else 0, axis=1)

    status('Pcalss and Sex')
    return dataset

dataset = process_pclass_sex()

##########
# Pclass #
##########
def process_pclass():
    global dataset
    # encoding into 3 categories:
    pclass_dummies = pd.get_dummies(dataset['Pclass'], prefix="Pclass")

    # adding dummy variable
    dataset = pd.concat([dataset, pclass_dummies], axis=1)

    # removing "Pclass"
    dataset.drop('Pclass', axis=1, inplace=True)

    status('Pclass')
    return dataset


dataset = process_pclass()

##########
# Ticket #
##########
"""
:param ticket ==> 
"""
def cleanTicket(ticket):
    # a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
    ticket = ticket.replace('.', '')
    ticket = ticket.replace('/', '')
    ticket = ticket.split()
    ticket = map(lambda x : x.strip(), ticket)
    ticket = list(filter(lambda t : not t.isdigit(), ticket))
    if len(ticket) > 0:
        return ticket[0]
    else:
        return 'X'

tickets = set()
for t in dataset['Ticket']:
    tickets.add(cleanTicket(t))

def process_ticket():
    global dataset

    # dummying
    dataset['Ticket'] = dataset['Ticket'].map(cleanTicket)
    tickets_dummies = pd.get_dummies(dataset['Ticket'], prefix='Ticket')
    dataset = pd.concat([dataset, tickets_dummies], axis=1)
    dataset.drop('Ticket', inplace=True, axis=1)

    status('Ticket')
    return dataset


dataset = process_ticket()

##########
# Family #
##########
def process_family():
    global dataset
    # introducing a new feature : the size of families (including the passenger)
    dataset['FamilySize'] = dataset['Parch'] + dataset['SibSp'] + 1

    # 가족 크기에 따라서.
    dataset['Singleton'] = dataset['FamilySize'].map(lambda x: 1 if x == 1 else 0)
    dataset['SmallFamily'] = dataset['FamilySize'].map(lambda x: 1 if 2 <= x <= 4 else 0)
    dataset['LargeFamily'] = dataset['FamilySize'].map(lambda x: 1 if 5 <= x else 0)

    status('family')
    return dataset


dataset = process_family()



# def main():
#     train = pd.read_csv('../data/train.csv')
#     outliers_indices = detect_outliers(train, 2, ["Age", "SibSp", "Parch", "Fare"])
#     print(outliers_indices)     # [27, 88, 159, 180, 201, 324, 341, 792, 846, 863]
#     print(len(outliers_indices))    # 10
#     print(dataset)
#
#
# if __name__ == '__main__':
#     main()