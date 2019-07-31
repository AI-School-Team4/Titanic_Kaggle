import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from danny.preprocessing_2 import dataset

def compute_score(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)
    return np.mean(xval)


def split_train_test_target():
    global dataset

    targets = pd.read_csv('../data/train.csv', usecols=['Survived'])['Survived'].values
    train = dataset.iloc[:881]
    test = dataset.iloc[881:]

    return train, test, targets


train, test, targets = split_train_test_target()

clf = RandomForestClassifier(n_estimators=150, max_features='sqrt')
clf = clf.fit(train, targets)

features = pd.DataFrame()
# Importance
# Age, Fare, Title_Mr, Sex, FamilySize....
features['feature'] = train.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)

# transform train and test in a more compact datasets
model = SelectFromModel(clf, prefit=True)
train_reduced = model.transform(train)
print(train_reduced.shape)  # (891, 13)

test_reduced = model.transform(test)
print(test_reduced.shape)   # (418, 13)

'''
Trying different base models
'''
# logreg = LogisticRegression()
# logreg_cv = LogisticRegressionCV()
# rf = RandomForestClassifier()
gboost = GradientBoostingClassifier()

models = [gboost]

for model in models:
    print('Cross-validation of : {0}'.format(model.__class__))
    score = compute_score(clf=model, X=train_reduced, y=targets, scoring='accuracy')
    print('CV score = {0}'.format(score))
    print('---------')

# Hyperparameters tuning
run_gs = True

if run_gs:
    parameter_grid = {
        'max_depth': [4, 6, 8],
        'n_estimators': [100, 50, 10],
        'max_features': ['sqrt', 'auto', 'log2'],
        'min_samples_split': [2, 3, 10],
        'min_samples_leaf': [1, 3, 10],
        'bootstrap': [True, False],
    }
    forest = RandomForestClassifier()
    cross_validation = StratifiedKFold(n_splits=5)

    grid_search = GridSearchCV(forest,
                               scoring='accuracy',
                               param_grid=parameter_grid,
                               cv=cross_validation,
                               verbose=1
                               )

    grid_search.fit(train, targets)
    model = grid_search.best_estimator_
    parameters = grid_search.best_params_

    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))

else:
    parameters = {'max_depth': 8, 'max_features': 'auto', 'min_samples_leaf': 10,
                  'min_samples_split': 2, 'n_estimators': 10}

    model = GradientBoostingClassifier(**parameters)
    model.fit(train, targets)



# Outputting
output = model.predict(test).astype(int)
df_output = pd.DataFrame()
test_data = pd.read_csv('../data/test.csv')
df_output['PassengerId'] = test_data['PassengerId']
df_output['Survived'] = output
df_output[['PassengerId','Survived']].to_csv('../submission/gboost_gs.csv', index=False)





