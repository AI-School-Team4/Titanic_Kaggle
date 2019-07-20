'''
preprocessing module
import convention : import appetiser as ap
'''
import numpy as np
import pandas as pd

# headers
# PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked

class load_titanic:
    ''' provide Titanic datasets and attributes '''

    def __init__(self):
        ''' useful attributes
                   train data    test data    type
        data     | self.data   | self.test  | DataFrame
        target   | self.target |            | Series
        col_size | self.size   | self.psize | int
        features |    self.all_features     | list_like (except for 'Survived')
        '''
        # File Path
        self.path_train = 'data/train.csv'
        self.path_test = 'data/test.csv'
        # DataFrame
        self.train = pd.read_csv(self.path_train)
        self.test = pd.read_csv(self.path_test)
        self.data = self.train.drop('Survived', axis=1)
        # Series
        self.target = self.train['Survived']
        self.PassengerId = self.test['PassengerId'] # PassengerId of test data (for submission)
        # int
        self.size = self.target.size
        self.psize = self.PassengerId.size
        # list
        self.all_features = self.test.columns
        self.features = None
    
    def select_features(self, features=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']):
        self.features = features

class dessert:
    ''' provide post-processing methods and attributes'''
    def __init__(self):
        pass

def check_missing_data(data, ratio=False):
    '''
    check how much missing data is
    data : DataFrame, Series
    ratio : bool
    return Series
    '''
    msdata = data.isnull().sum()
    if ratio:
        return msdata / data.iloc[:, 0].size
    else:
        return msdata

def error_not_series_with_na(data, skipna=False):
    if type(data) != type(pd.Series()):
        raise TypeError('Argument \'data\' must be Series')
    if (not skipna) and data.isnull().sum() != 0:
        raise ValueError('There is NA in argument \'data\'')

def one_hot_encoding(data):
    '''
    data : Series (without missing data, NA)
    return DataFrame
    '''
    # Expected Error Control
    error_not_series_with_na(data)
    
    # main algorithm
    uni = np.sort(data.unique())
    str_uni = ["{}_{}".format(data.name, name) for name in uni]
    ohe_df = pd.DataFrame(columns=str_uni, index=range(data.size))
    for irow, val in enumerate(data):
        icol = np.where(uni==val)[0][0]
        ohe_df.iloc[irow, :] = [1 if k==icol else 0 for k in range(uni.size)]
    return ohe_df

def complete_one_hot_encoding(data, columns):
    '''
    data : DataFrame
    columns : list
    return DataFrame
    '''
    for column in columns:
        ohe_df = one_hot_encoding(data[column])
        data = pd.concat([data, ohe_df], axis=1)
        data = data.drop(column, axis=1)
    return data

def normalize(data, skipna=False):
    '''
    data : Series
    skipna : bool (if True, normalizing ignoring NA)
    return Series
    XXX WARNING : sum of normalized data is not zero but small value
    '''
    # Expected Error Control
    error_not_series_with_na(data, skipna=skipna)

    # main algorithm
    data = (data - data.mean()) / data.std()
    return pd.Series(data.values, name='normalized_{}'.format(data.name), index=range(data.size))

def complete_normalization(data, columns):
    '''
    data : DataFrame
    columns : list
    return DataFrame
    '''
    for column in columns:
        n_df = normalize(data[column])
        data = pd.concat([data, n_df], axis=1)
        data = data.drop(column, axis=1)
    return data

def delete_na_row(data, target):
    '''
    data : DataFrame
    target : Series
    return DataFrame, Series (data, target)
    '''
    size = data.shape[0]
    index_to_remove = []
    for irow in range(size):
        for val in data.iloc[irow, :]:
            if pd.isna(val):
                index_to_remove.append(irow)
    data = data.drop(index_to_remove, axis=0)
    target = target.drop(index_to_remove, axis=0)
    return_data = pd.DataFrame(data.values, columns=data.columns, index=range(data.shape[0]))
    return_target = pd.Series(target.values, name=target.name, index=range(target.shape[0]))
    return return_data, return_target


def sort_columns(data):
    '''
    data : DataFrame (instance of class load_titanic)
    return DataFrame (sorted)
    '''
    order = {
        'PassengerId' : 0, 'Pclass' : 1, 'Name' : 2,
        'Sex' : 3, 'Age' : 4, 'SibSp' : 5, 'Parch' : 6,
        'Ticket' : 7, 'Fare' : 8, 'Cabin' : 9, 'Embarked' : 10
        }
    columns = data.columns
    # TODO : might need regular expression
    # TODO : incomplete
    for feature in data.all_features:
        extracted = pd.DataFrame(index=range(data.shape[0]))
        for column in columns:
            if feature in column:
                extracted = pd.concat([extracted, data[column]], axis=1)        
    pass


#===================================================
# TODO : methods below have to be modified
# def fill_value(data, value):
#     '''
#     data : Series, DataFrame (containing NA)
#     value : int, float, string, array_like
#     return Series, DataFrame
#     '''
#     return data.fillna(value)

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


# Temporary main method for debugging
# Should return pre-processed DataFrame
def main():
    pass

if __name__ == "__main__":
    main()

#========== DEBUG
    # a = one_hot_encoding(ex.data['Sex'])
    # print(a)
    # b = pd.read_csv("data/traintrain.csv")
    # for i in range(int(a.size / 2)):
    #     if a.iloc[i,0] == b.loc[i, 'Female'] and a.iloc[i,1]==b.loc[i,'Male']:
    #         pass
    #     else:
    #         print("Error")
    #         break
    # print("fine")
#==========
    # ex.data = pd.concat([ ex.data, one_hot_encoding(ex.data['Pclass']) ], axis=1)
    # ex.data = pd.concat([ ex.data, one_hot_encoding(ex.data['Sex']) ], axis=1)
    # ex.data = ex.data.drop(['Pclass', 'Sex', 'PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1)
    # ex.data.to_csv("data/preprocessing_train.csv", index=False)