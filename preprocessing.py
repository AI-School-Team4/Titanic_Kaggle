import pandas as pd
import numpy as np
from collections import Counter

# headers
# PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
from sklearn.model_selection import train_test_split

sex_mapping = {'male':0,'female':1}
title_mapping = {"Master": 0, "Miss": 1, "Ms": 1, "Mme": 1, "Mlle": 1, "Mrs": 1, "Mr": 2, "Rare": 3}
cabin_mapping = {'A':0,'B':0.4,'C':0.8,'D':1.2,'E':1.6,'G':2.4,'T':2.8}
embarked_mapping = {'S':0,'C':1,'Q':2}

'''
NA inclued features:
    Age, Cabin, Embarked
'''

def preprocessor(train, test, fill_age_with, fill_cabin_with, dropPassengerID=True, dropName=True, dropTicket=True):
    """
    @:param
    (DataFrame) train DF
    (DataFrame) test DF
    (String) fill_age_with => {median, advanced_median, something_else}
    (bool) dropPassengerID => default == False
    (bool) dropName        => default == False

    <Cabin>
    1. 많이 비어 있기 때문에 채워줘야 할듯
    2. String 인덱스 0번 값 기준으로 새로운 시리즈 추가

    <Embarked>
    1. 한두개만 비어 있음. 어떻게 해야할까

    """

    train_len = len(train)
    # Concatenate train and test dataFrame
    dataset = pd.concat(objs=[train, test], axis=0).reset_index(drop=True)

    # Fill empty and NaNs values with NaN
    dataset = dataset.fillna(np.nan)

    ########
    # Fare #
    ########
    # scaling Fare by applying log
    dataset["Fare"] = dataset["Fare"].map(lambda x: np.log(x) if x > 0 else 0)

    # TODO : 로그스케일로 바꿨을때 Binning을 한다면 그의 범위를 어떻게 나누면 좋을까

    # for data in dataset:
    #     data.loc[data['Fare'] <= 17, 'Fare'] = 0,
    #     data.loc[(data['Fare'] > 17) & (data['Fare'] <= 30), 'Fare'] = 1,
    #     data.loc[(data['Fare'] > 30) & (data['Fare'] <= 100),'Fare'] = 2,
    #     data.loc[data['Fare'] > 100, 'Fare'] = 3


    ########
    # Sex  #
    ########
    # convert Sex into dict val 0 for male and 1 for female
    dataset["Sex"] = dataset["Sex"].map(sex_mapping)


    ########
    # Name #
    ########
    # Extract Title from Name
    extracted_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]
    dataset["Title"] = pd.Series(extracted_title)

    # Convert to dict val; Title
    dataset["Title"] = dataset["Title"].replace(
        ['Lady', 'the Countess', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],
        'Rare')

    # unique titles
    # ['Mr', 'Mrs', 'Miss', 'Master', 'Don', 'Rev', 'Dr', 'Mme', 'Ms',
    #  'Major', 'Lady', 'Sir', 'Mlle', 'Col', 'Capt', 'Countess', 'Jonkheer']

    dataset["Title"] = dataset["Title"].map(title_mapping).astype(int)

    if dropName:
        # Drop Name variable
        dataset.drop(labels=["Name"], axis=1, inplace=True)


    ########
    # Age  #
    ########
    # Index of NaN age rows
    index_to_fill_age = list(dataset["Age"][dataset["Age"].isnull()].index)

    if fill_age_with == 'median':
        # normal median
        for i in index_to_fill_age:
            median_age = dataset["Age"].median()
            dataset['Age'].iloc[i] = median_age

    elif fill_age_with == 'advanced_median_1':
        # Fill Age with the median age of similar rows according to Pclass, Parch and SibSp
        for i in index_to_fill_age:
            age_med = dataset["Age"].median()
            age_pred = dataset["Age"][((dataset['SibSp'] == dataset.iloc[i]["SibSp"]) & (dataset['Parch'] == dataset.iloc[i]["Parch"]) & (dataset['Pclass'] == dataset.iloc[i]["Pclass"]))].median()
            if not np.isnan(age_pred):
                dataset['Age'].iloc[i] = age_pred
            else :
                dataset['Age'].iloc[i] = age_med

    # TODO : check if it's efficient enough
    elif fill_age_with == 'advanced_median_2':
        dataset['Age'].fillna(dataset.groupby('Title')['Age'].transform('median'), inplace=True)


    #########
    # Cabin #
    #########
    if fill_cabin_with == 'X':
        # Replace the Cabin number by the type of cabin 'X' if not
        dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin']])
        dataset = pd.get_dummies(dataset, columns = ["Cabin"],prefix="Cabin")

    elif fill_cabin_with == 'mapping_median':
        # Cabin mapping
        dataset['Cabin'] = dataset['Cabin'].str[:1]
        dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)

        # Fill with median
        # Todo : dangerous???
        dataset['Cabin'].fillna(dataset.groupby('Pclass')['Cabin'].transform('median'), inplace=True)


    ###############
    # PassengerID #
    ###############
    if dropPassengerID:
        # Drop useless variables
        dataset.drop(labels = ["PassengerId"], axis = 1, inplace = True)


    ##########
    # Family #
    ##########
    # Create a family size descriptor from SibSp and Parch
    dataset["Fsize"] = dataset["SibSp"] + dataset["Parch"] + 1
    # Create new feature of family size
    dataset['Single'] = dataset['Fsize'].map(lambda x: 1 if x == 1 else 0)
    dataset['SmallF'] = dataset['Fsize'].map(lambda x: 1 if x == 2 else 0)
    dataset['MedF'] = dataset['Fsize'].map(lambda x: 1 if 3 <= x <= 4 else 0)
    dataset['LargeF'] = dataset['Fsize'].map(lambda x: 1 if x >= 5 else 0)

    ############
    # Embarked #
    ############
    '''
    More than half of the first, second and third class people are come from S.
    So fill with S.
    '''
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)

    ###########
    # Ticket  #
    ###########
    if dropTicket:
        dataset.drop(labels=["Ticket"], axis=1, inplace=True)

    else:
        # Idea: Extract ticket prefix.
        # When there is no prefix replace with X
        Ticket = []
        for i in list(dataset.Ticket):
            if not i.isdigit():
                Ticket.append(i.replace(".", "").replace("/", "").strip().split(' ')[0])  # Take prefix
            else:
                Ticket.append("X")

        dataset["Ticket"] = pd.Series(Ticket)

    ###########
    # Dummies #
    ###########
    dataset = pd.get_dummies(dataset, columns=["Title"])
    dataset = pd.get_dummies(dataset, columns=["Embarked"], prefix="Em")
    # dataset = pd.get_dummies(dataset, columns=["Cabin"], prefix="Cabin")
    # dataset = pd.get_dummies(dataset, columns=["Ticket"], prefix="T")
    # dataset["Pclass"] = dataset["Pclass"].astype("category")
    # dataset = pd.get_dummies(dataset, columns=["Pclass"], prefix="Pc")

    ###########
    # The End #
    ###########
    # Split concatenated train and test DataFrame
    train = dataset[:train_len]
    test = dataset[train_len:]
    test.drop(labels=["Survived"], axis=1, inplace=True)

    # Separate train features and label
    train["Survived"] = train["Survived"].astype(int)
    X_train = train.drop(labels=["Survived"], axis=1)
    Y_train = train["Survived"]

    return X_train, Y_train, test


# Temporary main method for debugging
# Should return pre-processed DataFrame

# def main():
#     train_dataset = pd.read_csv('data/train.csv')
#     test_dataset = pd.read_csv('data/test.csv')
#
#     X_train_processed, Y_train_processed, test_processed = preprocessor(train_dataset, test_dataset,
#                                                                         fill_age_with='median',
#                                                                         fill_cabin_with='mapping_median',
#                                                                         dropPassengerID=True, dropName=True)
#
#     X_train, X_valid, y_train, y_valid = train_test_split(X_train_processed, Y_train_processed, test_size=0.2,
#                                                           random_state=np.random.seed())
#
#
# if __name__ == "__main__":
#     main()

