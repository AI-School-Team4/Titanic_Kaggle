# preprocessing module
import numpy as np
import pandas as pd

# headers
# PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
class Titanic:
    '''
    docstring
    '''
    path_train = 'data/train.csv'
    path_test_data = 'data/test.csv'
    path_test_target = 'data/gender_submission.csv'

    def __init__(self):
        # DataFrame
        self.train = pd.read_csv(Titanic.path_train)
        self.test = pd.read_csv(Titanic.path_test_data)
        self.plabel = pd.read_csv(Titanic.path_test_target) # label with PassengerId (DataFrame)
        self.data = self.train.drop('Survived', axis=1)
        # Series
        self.target = self.train['Survived']
        self.label = self.plabel['Survived']
        # int
        self.size = self.target.size
        self.sizel = self.label.size
        # list
        self.features = self.test.columns


    def fill_na_with_median(self):
        pass

    def delete_na_row(self):
        #
        data_ = self.data.copy()
        index_to_remove = []
        for irow in range(self.size):
            for val in data_[irow]:
                if pd.isna(val):
                    index_to_remove.append(irow)
        return data_.drop(index_to_remove, axis=0)

#==================================================

def load_train_data():
    # DataFrame
    return pd.read_csv('data/train.csv')

def load_test_data():
    # DataFrame
    return pd.read_csv('data/test.csv')

def load_test_target(dtype='Series'): # dtype == 'Series' or 'DataFrame'
    test_target = pd.read_csv('data/gender_submission.csv')
    # Series
    if dtype == 'Series' or dtype == 'series':
        return test_target['Survived']
    # DataFrame
    elif dtype == 'DataFrame' or dtype == 'dataframe':
        return test_target
    # Error
    else:
        raise ValueError('Invalid dtype')
        return None

def fill_median(data):
    # type(data) == DataFrame, dtype == int, float
    size = int(data.size / len(data.columns))
    
    for icol in range(len(data.columns)):
        index_to_fill = []
        value = []
        for irow in range(size):
            val = data.iloc[irow, icol]
            if pd.isnull(val):
                index_to_fill.append(irow)
            else:
                value.append(val)
        median = np.median(value)
        for irow in index_to_fill:
            data.iloc[irow, icol] = median
    return data

def fill_mean(data):
    # type(data) == DataFrame, dtype == int, float
    size = int(data.size / len(data.columns))
    
    for icol in range(len(data.columns)):
        index_to_fill = []
        value = []
        for irow in range(size):
            val = data.iloc[irow, icol]
            if pd.isnull(val):
                index_to_fill.append(irow)
            else:
                value.append(val)
        mean = np.mean(value)
        for irow in index_to_fill:
            data.iloc[irow, icol] = mean
    return data

def delete_na_row(data):
    # type(data) == DataFrame
    size = int(data.size / len(data.columns))
    index_to_remove = []
    for irow in range(size):
        for val in data.iloc[irow, :]:
            if pd.isna(val):
                index_to_remove.append(irow)
    return data.drop(index_to_remove, axis=0)

def data_target_split(unseparated):
    data = unseparated.drop('Survived', axis=1)
    target = unseparated['Survived']
    return data, target

# Temporary main method for debugging
# Should return pre-processed DataFrame
def main():
    a = delete_na_row(load_train_data())
    print(a)

if __name__ == "__main__":
    main()