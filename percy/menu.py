'''
drawing module
import convention : import menu
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import percy.appetiser as ap


def missing_matrix(data, figsize=(8,8), color=(0.8,0.5,0.2)):
    ''' visualizing missing data '''
    msno.matrix(df=data.iloc[:,:], figsize=figsize, color=color)
    plt.show()

def distribution_of(data, col):
    ''' distribution of data[col] without missing data '''
    sns.distplot(data[col].dropna())
    plt.show()

def graph_2columns(data, col1, col2):
    ''' NA '''
    # TODO : should add legend
    # reference : https://datascienceschool.net/view-notebook/8cbbdd4daaf84c0492d440b9a819c8be/
    graph = sns.FacetGrid(data, hue=col1, height=4, aspect=2)
    graph.map(sns.kdeplot, col2, legend=True)
    plt.show()

def show_correlation(data):
    '''
    data : DataFrame
    '''
    heatmap_data = data.loc[:,:]
    colormap = plt.cm.RdBu
    plt.figure(figsize=(7,6))
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sns.heatmap(heatmap_data.astype(float).corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True, annot_kws={"size":16})
    plt.show()
    del heatmap_data

def main():
    pass

if __name__ == "__main__":
    main()
    pass