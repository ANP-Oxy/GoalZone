'''
This file contains all the utility functions I used
in the jupyter Notebook containing GoalZone case study
'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def validate(df, data_dict):
    '''
    This function takes a dataframe and validates
    it against the in built data dictionary
    '''
    for type in ['number', 'category']:

        # first check if numeric columns have values
        # above minimum specified value
        if type == 'number':
            for column in df.select_dtypes('number').columns:
                try:
                    assert min(df[column]) >= data_dict[column]
                    print(f'"{column}" passed the validation test')
                except AssertionError:
                    print(f'"{column}" failed the validation test')

        # Check if the categorical columns have valid
        # values as per data dictionary
        else:
            for column in df.select_dtypes('category').columns:
                try:
                    assert set(df[column]) == set(data_dict[column])
                    print(f'"{column}" passed the validation test')
                except AssertionError:
                    print(f'"{column}" failed the validation test')


def numerics(df, labels):

    '''
    This function returns boxplots to show relationship
    between numerical features and target variable
    '''
    fig, ax = plt.subplots(1, 3, figsize=(12, 6))
    cols = [col for col in df.select_dtypes('number')]
    num = 0
    for col in cols:
        sns.boxplot(data=df, x=labels, y=col, ax=ax[num])
        num += 1


def categoricals(df, labels):
    '''
    This function returns how proportion of target variables
    is distributed for various categorical features.
    '''
    cat_cols = [col for col in df.select_dtypes('category')]
    fig, ax = plt.subplots(len(cat_cols), 1, figsize=(12, 10))
    row = 0
    for col in cat_cols:
        (df
         .groupby(labels)
         [col]
         .value_counts(normalize=True)
         .unstack()
         .plot
         .bar(ax=ax[row]))
        row += 1


def get_relations(dataframe, y, which):

    '''
    This function takes a dataframe and returns matplotlib
    plots for checking relationship of features with target variables

    parameters
    dataframe: a pandas dataframe
    which: "numerics" or "categoricals" calls either of these functions. 
    '''
    # replacing target variable [1, 0] with more readable "Yes", "No"
    labels = y.replace({0: "No", 1: "Yes"})
    if which == "numerics":
        numerics(df=dataframe, labels=labels)
    if which == "categoricals":
        categoricals(df=dataframe, labels=labels)


def outlier_rejection(X, y):
    '''
    This function takes two arrays X and y 
    and returns the arrays with outliers removed.
    the outliers are considered to be the data points
    outside of three standard deviation mark
    '''
    X_, y_ = X, y
    train = np.hstack([X_, y_.reshape(-1, 1)])
    for i in [0, -3, -2]:
        Z_scores = np.abs((train[:, i].mean() - train[:, i]) /
                          train[:, i].std())
        train = train[Z_scores < 3]
    X_new = train[:, :-1]
    y_new = train[:, -1]
    return X_new, y_new
