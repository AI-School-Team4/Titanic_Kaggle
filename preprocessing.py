import numpy as np
import pandas as pd

# headers
# PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked

def load_train_data():
    # DataFrame
    return pd.read_csv('data/train.csv')

def load_test_data():
    # DataFrame
    return pd.read_csv('data/test.csv')


def fill_na_with_median():
    pass

def delete_na_row():
    pass



# Temporary main method for debugging
# Should return pre-processed DataFrame
def main():
    train = load_train_data()
    print(train.head())

if __name__ == "__main__":
    main()