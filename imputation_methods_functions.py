#!/usr/bin/env python3

import pandas as pd
import numpy as np
import copy
from sklearn.model_selection import train_test_split

### IMPUTATION METHODS FUNCTIONS ###

## ZEROS ##
# Zeros - can be used for both training and test set
def zeros(dataframe):
    new_dataframe = dataframe.fillna(0)
    
    return new_dataframe


## MAJORITY ##
# Majority - on training set
def majority_train(dataframe_train):
    
    new_dataframe = dataframe_train.fillna(dataframe_train.mode().iloc[0])
    
    return new_dataframe


# Majority - on test set (based on a training set)
def majority_test(dataframe_train, dataframe_test):
    
    # If the input is a dataframe
    if isinstance(dataframe_test, pd.DataFrame):
        # Impute each feature
        for feature in dataframe_train:
            # Majority value for this feature (in training set)
            majority_value = dataframe_train[feature].mode()[0]
            
            # Fill in this value for all NAs in this feature (in test set)
            dataframe_test[feature][np.isnan(dataframe_test[feature])] = majority_value
    
    # If the input is an array (then we know it is binary because it is an outcome column)
    else:
        dataframe_test[np.isnan(dataframe_test)] = dataframe_train.mode()[0]
    
    return dataframe_test


## COMBINATIONS ##
# Zeros, majority - on training set
def zeros_majority_train(dataframe):
    new_dataframe = copy.deepcopy(dataframe)
    
    # If the input is a dataframe
    if isinstance(dataframe, pd.DataFrame):
        binary_cols = [col for col in dataframe if np.isin(dataframe[col].dropna().unique(), [0, 1]).all()]
        non_binary_cols = [col for col in dataframe if col not in binary_cols]
        
        # Zeros
        new_dataframe[binary_cols] = new_dataframe[binary_cols].fillna(0)
        
        # Majority
        new_dataframe[non_binary_cols] = new_dataframe[non_binary_cols].fillna(dataframe.mode().iloc[0])
    
    # If the input is an array (then we know it is binary because it is an outcome column)
    else:
        new_dataframe[np.isnan(new_dataframe)] = 0
        
    return new_dataframe



# Zeros, majority - on test set (based on training set)
def zeros_majority_test(dataframe_train, dataframe_test):
    new_dataframe = copy.deepcopy(dataframe_test)
    
    # If the input is a dataframe
    if isinstance(dataframe_test, pd.DataFrame):
        binary_cols = [col for col in dataframe_test if np.isin(dataframe_test[col].dropna().unique(), [0, 1]).all()]
        non_binary_cols = [col for col in dataframe_test if col not in binary_cols]
        
        # Fill binary columns with zeros
        new_dataframe[binary_cols] = new_dataframe[binary_cols].fillna(0)
    
        # Fill non-binary columns with majority (from training set)
        for feature in dataframe_test[non_binary_cols]:
            # Majority value for this feature (in training set)
            majority_value = dataframe_train[feature].mode()[0]
            
            # Fill in this value for all NAs in this feature (in test set)
            new_dataframe[feature][np.isnan(new_dataframe[feature])] = majority_value
    
    # If the input is an array (then we know it is binary because it is an outcome column)
    else:
        new_dataframe[np.isnan(new_dataframe)] = 0
        
    return new_dataframe




# Majority, median - on training set
def majority_median_train(dataframe):
    new_dataframe = copy.deepcopy(dataframe)
    
    # If the input is a dataframe
    if isinstance(dataframe, pd.DataFrame):
        binary_cols = [col for col in dataframe if np.isin(dataframe[col].dropna().unique(), [0, 1]).all()]
        non_binary_cols = [col for col in dataframe if col not in binary_cols]
    
        # Majority
        new_dataframe[binary_cols] = new_dataframe[binary_cols].fillna(dataframe.mode().iloc[0])
    
        # Median
        new_dataframe[non_binary_cols] = new_dataframe[non_binary_cols].fillna(dataframe.median())

    
    # If the input is an array (then we know it is binary because it is an outcome column)
    else:
        new_dataframe[np.isnan(new_dataframe)] = dataframe.mode().iloc[0]
        
    return new_dataframe


# Majority, median - on test set (based on training set)
def majority_median_test(dataframe_train, dataframe_test):
    new_dataframe = copy.deepcopy(dataframe_test)
    
    # If the input is a dataframe
    if isinstance(dataframe_test, pd.DataFrame):
        binary_cols = [col for col in dataframe_test if np.isin(dataframe_test[col].dropna().unique(), [0, 1]).all()]
        non_binary_cols = [col for col in dataframe_test if col not in binary_cols]
        
        # Majority
        for feature in dataframe_test[binary_cols]:
            # Majority value for this feature (in training set)
            majority_value = dataframe_train[feature].mode()[0]
            
            # Fill in this value for all NAs in this feature (in test set)
            new_dataframe[feature][np.isnan(new_dataframe[feature])] = majority_value
            
        # Median
        for feature in dataframe_test[non_binary_cols]:
            # Median value for this feature (in training set)
            median_value = dataframe_train[feature].median()
            
            # Fill in this value for all NAs in this feature (in test set)
            new_dataframe[feature][np.isnan(new_dataframe[feature])] = median_value
    
    # If the input is an array (then we know it is binary because it is an outcome column)
    else:
        new_dataframe[np.isnan(new_dataframe)] = dataframe_train.mode().iloc[0]
        
    return new_dataframe






