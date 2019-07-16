import numpy as np
import pandas as pd

# headers
# PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked

def load_train_data():
    return pd.read_csv('data/train.csv')

def load_test_data():
    return pd.read_csv('data/test.csv')




def main():
    train = load_train_data()
    print(train)

if __name__ == "__main__":
    main()