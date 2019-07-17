import pandas as pd
import numpy as np
from collections import Counter

# headers
# PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked

def fill_na_with_median():
    pass

def data_target_split(unseparated):
    unseparated["Survived"] = unseparated["Survived"].astype(int)
    data = unseparated.drop('Survived', axis=1)
    target = unseparated['Survived']
    return data, target

# Undesirable
def delete_na_row():
    # Cant delete 'Cabin' Na rows. too many
    pass


'''
NA inclued features:
    Age, Cabin, Embarked
'''


def preprocessor(dataset, fill_age_with, dropPassengerID=False, dropName=False):
    """
    @:param
    (DataFrame) dataset
    (String) fill_age_with => {median, advanced_median, something_else}
    (bool) dropPassengerID => default == False
    (bool) dropName        => default == False

    <Cabin>
    1. 많이 비어 있기 때문에 채워줘야 할듯
    2. String 인덱스 0번 값 기준으로 새로운 시리즈 추가

    <Embarked>
    1. 한두개만 비어 있음. 어떻게 해야할까

    """

    train_len = len(dataset)

    # Fill empty and NaNs values with NaN
    dataset = dataset.fillna(np.nan)

    ########
    # Fare #
    ########
    # scaling Fare
    # Apply log to Fare to reduce skewness distribution
    dataset["Fare"] = dataset["Fare"].map(lambda i: np.log(i) if i > 0 else 0)

    ########
    # Sex  #
    ########
    # convert Sex into categorical value 0 for male and 1 for female
    dataset["Sex"] = dataset["Sex"].map({"male": 0, "female":1})

    ########
    # Age  #
    ########
    # Index of NaN age rows
    index_NaN_age = list(dataset["Age"][dataset["Age"].isnull()].index)

    # TODO : normal median
    if fill_age_with == 'median':
        pass

    elif fill_age_with == 'advanced_median':
        # Fill Age with the median age of similar rows according to Pclass, Parch and SibSp
        for i in index_NaN_age:
            age_med = dataset["Age"].median()
            age_pred = dataset["Age"][((dataset['SibSp'] == dataset.iloc[i]["SibSp"]) & (dataset['Parch'] == dataset.iloc[i]["Parch"]) & (dataset['Pclass'] == dataset.iloc[i]["Pclass"]))].median()
            if not np.isnan(age_pred) :
                dataset['Age'].iloc[i] = age_pred
            else :
                dataset['Age'].iloc[i] = age_med

    # TODO : what would be the other possible metrics
    elif fill_age_with == 'something_else':
        pass

    ########
    # Name #
    ########
    if dropName:
        # Drop Name variable
        dataset.drop(labels=["Name"], axis=1, inplace=True)

    #########
    # Cabin #
    #########
    # Replace the Cabin number by the type of cabin 'X' if not
    dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin'] ])
    dataset = pd.get_dummies(dataset, columns = ["Cabin"],prefix="Cabin")


    ###############
    # PassengerID #
    ###############
    if dropPassengerID:
        # Drop useless variables
        dataset.drop(labels = ["PassengerId"], axis = 1, inplace = True)
        dataset = dataset[:train_len]

    ############
    # Embarked #
    ############
    # TODO : ideas?
    pass

    ###########
    # Ticket  #
    ###########
    # TODO : ideas?
    pass

    # Separate train features and label
    dataset["Survived"] = dataset["Survived"].astype(int)
    X_train = dataset.drop(labels=["Survived"], axis=1)
    Y_train = dataset["Survived"]

    return X_train, Y_train


# Temporary main method for debugging
# Should return pre-processed DataFrame
# def main():
#     train = load_train_data()
#
#     # 'Age', 'Cabin' and 'Embarked' have na values
#     for col in train:
#         if train[col].isna().sum() != 0:
#             print(col)
#
# if __name__ == "__main__":
#     main()

