'''
main module
'''
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import percy.appetiser as ap
import percy.maindish as md
import percy.menu
from percy.appetiser import load_titanic
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# preprocessing data 1
'''
instance name : ex
data name : apple
selected features : Pclass |  Sex  |  Age  | SibSp | Parch | Fare | Embarked
normalizaion      :                    O                       O
one-hot-encoding  :    O       O                                       O
missing-data      :                 removed                          removed
'''
ex = load_titanic()
ex.select_features() # ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
apple = ex.data[ex.features]
# menu.graph_2columns(apple, 'Pclass', 'Age')
apple, label_apple = ap.delete_na_row(apple, ex.target)
apple = ap.complete_one_hot_encoding(apple, ['Pclass', 'Sex', 'Embarked'])
apple = ap.complete_normalization(apple, ['Fare', 'Age'])
# print(apple)
# menu.show_correlation(apple)
apple.to_csv("data/preprocessing_train.csv", index=False)

label_apple = ap.one_hot_encoding(label_apple)
label_apple.to_csv("data/target.csv", index=False)

# hyperparameter control
inodes = 12
hnodes = 300
hnumber = 2
onodes = 2
eta = 0.05
epoch = 15

model_apple = md.BinaryClassification(inodes, hnodes, hnumber, onodes, eta)
for e in range(epoch):
    model_apple.fit(apple, label_apple)
predicted = model_apple.predict(apple)
accuracy = model_apple.accuracy_score(label_apple, predicted)

print(predicted)
print('accuracy : {0} ==> {1:0.2f}%'.format(accuracy, accuracy * 100))

dt_clf = DecisionTreeClassifier()
dt_clf.fit(apple, label_apple)
predicted = dt_clf.predict(apple)
accuracy = accuracy_score(label_apple, predicted)
print('accuracy from sklearn DecisionTreeClassifier : {0:0.2f}%'.format(accuracy * 100))