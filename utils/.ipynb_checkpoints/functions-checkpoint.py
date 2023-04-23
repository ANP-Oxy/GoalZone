'''
This file contains all the utility functions I used in the jupyter Notebook containing GoalZone case study
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import urllib


def read_data(url, filename=None):
    '''
    This function takes a csv file url and the file name of 
    the file to be retrieved
    returns a pandas DataFrame of the csv
    '''
    urllib.request.urlretrieve(url, filename = filename)
    return pd.read_csv(url)





def validate(df, data_dict):
    '''
    This function takes a dataframe and validates it against the in built data dictionary
    '''
    
    for type in ['number', 'category']:

        # first check if numeric columns have values above minimum specified value
        if type == 'number':
            for column in df.select_dtypes('number').columns:
                try:
                    assert min(df[column]) >= data_dict[column]
                    print(f'"{column}" passed the validation test')
                except AssertionError:
                    print(f'"{column}" failed the validation test')

        # Check if the categorical columns have valid values as per data dictionary
        else:
            for column in df.select_dtypes('category').columns:
                try:
                    assert set(df[column]) == set(data_dict[column])
                    print(f'"{column}" passed the validation test')
                except AssertionError:
                    print(f'"{column}" failed the validation test')
                    
                        

 


def numerics(*args):

    '''
    This function returns boxplots to show relationship between numerical features and target variable. 
    '''
    fig, ax = plt.subplots(1, 3, figsize=(12,6))
    cols = [col for col in dataframe.select_dtypes('number')]
    num = 0    
    for col in cols:
        sns.boxplot(data=dataframe, x=labels, y=col, ax=ax[num])
        num+= 1



def categoricals(*args):
    '''
    This function returns how proportion of target variables is distributed for various categorical features. 
    '''
    cat_cols = [col for col in dataframe.select_dtypes('category')]
    
    fig, ax = plt.subplots(len(cat_cols), 1, figsize=(12, 10))
    row = 0
    
    for col in cat_cols:
        (dataframe
         .groupby(labels)
         [col]
         .value_counts(normalize=True)
         .unstack()
         .plot
         .bar(ax=ax[row]))
        row+= 1



def get_relations(dataframe, y, which):

    '''
    This function takes a dataframe and returns matplotlib plots for checking relationship of features with target variables

    parameters
    dataframe: a pandas dataframe
    which: "numerics" or "categoricals" calls either of these functions. 
    '''
    # replacing target variable [1, 0] with more readable "Yes", "No"
    labels = y.replace({0:"No", 1:"Yes"})
    
    if which == "numerics":
        numerics(dataframe, labels)
        
    if which == "categoricals":
        categoricals(dataframe, labels)




