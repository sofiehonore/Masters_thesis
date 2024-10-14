#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import imputation_methods_functions as imputation_methods
import sys
import ast
import subprocess
import copy

## Get outcome from input ##
outcome = sys.argv[1]
imputation_method = sys.argv[2]

## Load data ##
df = pd.read_csv('/home/projects/ssi_10004/data/updated_data_one_hot_NA_threshold.csv', sep = ";", decimal = ",")

# Define outcome and delete all outcome columns
outcome_col = df[outcome]

df = df.drop(['hard_neo', 'moderate_neo', 'soft_neo', 'hard_mat', 'moderate_mat', 'soft_mat',
         'outcome_hard_neo', 'outcome_hard_mod_neo', 'outcome_any_neo', 'outcome_hard_mat', 'outcome_hard_mod_mat', 'outcome_any_mat',
         'outcome_hard_mat_neo', 'outcome_hard_mod_mat_neo', 'outcome_any_mat_neo'], axis = 1)

# Define imputation methods to use
if imputation_method == "zeros":
    imputation_method_train = imputation_methods.zeros
    imputation_method_test = imputation_methods.zeros
elif imputation_method == "majority":
    imputation_method_train = imputation_methods.majority_train
    imputation_method_test = imputation_methods.majority_test
elif imputation_method == "zeros_majority":
    imputation_method_train = imputation_methods.zeros_majority_train
    imputation_method_test = imputation_methods.zeros_majority_test
elif imputation_method == "majority_median":
    imputation_method_train = imputation_methods.majority_median_train
    imputation_method_test = imputation_methods.majority_median_test


## CV ##
# Split into five data sets that can take turns being test and train/validation
number_of_sets = 5
df_split = np.array_split(df, number_of_sets)
outcome_split = np.array_split(outcome_col, number_of_sets)

set_list = list(range(number_of_sets))

# Dicts
AUCs_all_features = dict()
feature_importances_all_features = dict()
permutation_importances_all_features = dict()

# Number of runs when finding permutation importances
n_runs_permutation_importance = 20

### AUC AND RF FEATURE IMPORTANCE ###
for t_set in range(number_of_sets):
    print("TEST SET:", t_set)
    # Define test set for this round 
    X_test = df_split[t_set]
    y_test = outcome_split[t_set]
    
    # Remaining sets
    remaining_sets = set_list[:t_set] + set_list[t_set+1:]
    
    # df split without test set
    df_split_array = np.array(df_split, dtype = object)
    df_split_wo_test = (list(df_split_array[remaining_sets]))
    
    # outcome without test set
    outcome_split_array = np.array(outcome_split, dtype = object)
    outcome_split_wo_test = (list(outcome_split_array[remaining_sets]))
    
    # Subdicts for storing AUCs and feature importances for this test set
    AUCs_all_features[t_set] = dict()
    feature_importances_all_features[t_set] = dict()
    permutation_importances_all_features[t_set] = dict()
    
    # Define full training set (everything except test set)
    X_train = pd.concat(df_split_wo_test)
    y_train = pd.concat(outcome_split_wo_test)
    
    # Impute full training set
    X_train_impute = imputation_method_train(X_train)
    y_train_impute = imputation_method_train(y_train)
    
    # Impute test set based on full training set
    if imputation_method == "zeros":
        X_test_impute = imputation_method_test(X_test)
        y_test_impute = imputation_method_test(y_test)
    else:
        X_test_impute = imputation_method_test(X_train, X_test)
        y_test_impute = imputation_method_test(y_train, y_test)
    
    ## FEATURE IMPORTANCES ##
    # Run RF on this training set with all features and default hyperparameters
    clf = RandomForestClassifier()
    clf.fit(X_train_impute, y_train_impute)
    
    feature_importances = clf.feature_importances_
    
    # Define feature number for index in feature_importances
    feature_number = 0
    
    ## RUN THROUGH ALL FEATURES ##
    for feature in X_train_impute:
        ## AUCs ##
        # Define predictor column
        pred = X_train_impute[feature]
        outcome = y_train_impute
        
        # Combine outcome and predictor and write to file
        roc_file = pd.concat([outcome, pred], axis = 1)
        
        # File name    
        file_name = str('/home/projects/ssi_10004/people/sofhor/roc_files/roc_file_' + feature + '.csv')
        
        # Write pred and measure to files
        roc_file.to_csv(file_name, sep = "\t", header = None, index = False)
        
        # Run roc on the resulting file
        job = subprocess.run(['./roc', '-pc', file_name], stdout=subprocess.PIPE, universal_newlines=True)
        lines = job.stdout.split('\n')
        
        # Extract AUC and add to dict
        for line in lines:
            if line.startswith("# AUC"):
                AUC = float(line.split()[2])
                
                if AUC < 0.5:
                    AUC = 1 - AUC
                
                AUCs_all_features[t_set][feature] = AUC
        
        ## SAVE RF FEATURE IMPORTANCE FOR THIS FEATURE ##
        feature_importances_all_features[t_set][feature] = feature_importances[feature_number]
        feature_number += 1
        
        ## CREATE LIST FOR STORING PERMUTATION IMPORTANCES LATER ##
        permutation_importances_all_features[t_set][feature] = list()
        


### PERMUTATION FEATURE IMPORTANCE ###
for t_set in range(number_of_sets):
    print("TEST SET:", t_set)
    
    # Define test set for this round 
    X_test = df_split[t_set]
    y_test = outcome_split[t_set]
    
    # Remaining sets
    remaining_sets = set_list[:t_set] + set_list[t_set+1:]
    
    # df split without test set
    df_split_array = np.array(df_split, dtype = object)
    df_split_wo_test = (list(df_split_array[remaining_sets]))
    
    # outcome without test set
    outcome_split_array = np.array(outcome_split, dtype = object)
    outcome_split_wo_test = (list(outcome_split_array[remaining_sets]))
        
    for v_set in range(len(remaining_sets)):
        # Define validation set
        X_validation = df_split_wo_test[v_set]
        y_validation = outcome_split_wo_test[v_set]
        
        # Create training set for this round
        if v_set == 0:
            X_train = pd.concat(df_split_wo_test[v_set+1:])
            y_train = pd.concat(outcome_split_wo_test[v_set+1:])
        elif v_set == number_of_sets - 2: # because test set has been taken out
            X_train = pd.concat(df_split_wo_test[:v_set])
            y_train = pd.concat(outcome_split_wo_test[:v_set])
        else:
            X_train1 = pd.concat(df_split_wo_test[:v_set])
            X_train2 = pd.concat(df_split_wo_test[v_set+1:])
            
            y_train1 = pd.concat(outcome_split_wo_test[:v_set])
            y_train2 = pd.concat(outcome_split_wo_test[v_set+1:])
            
            X_train = pd.concat([X_train1, X_train2])
            y_train = pd.concat([y_train1, y_train2])
        
        # Impute training set and validation set
        X_train_impute = imputation_method_train(X_train)
        y_train_impute = imputation_method_train(y_train)
        
        if imputation_method == "zeros":
            X_validation_impute = imputation_method_test(X_validation)
            y_validation_impute = imputation_method_test(y_validation)
        else:
            X_validation_impute = imputation_method_test(X_train, X_validation)
            y_validation_impute = imputation_method_test(y_train, y_validation)
            
        
        ## GET BASELINE AUC FOR THIS VALIDATION SET ##
        # Run RF to get baseline model for this validation set
        clf = RandomForestClassifier()
        clf.fit(X_train_impute,y_train_impute)
        
        # Get prediction of RF with these chosen features
        y_pred = clf.predict_proba(X_validation_impute)[:,1]
        
        # Combine outcome and predictor and write to file
        roc_df = pd.DataFrame()
        roc_df['Validation'] = y_validation_impute.reset_index(drop = True)
        roc_df['Prediction'] = y_pred
        
        # File name 
        file_name = '/home/projects/ssi_10004/people/sofhor/roc_files/roc_file_permutation_importances_baseline_t' + str(t_set) + '_v' + str(v_set) + '.csv'
        
        # Write y pred and y validation to file
        roc_df.to_csv(file_name, sep = "\t", header = None, index = False)
        
        # Run roc on the resulting file
        job = subprocess.run(['./roc', '-pc', file_name], stdout=subprocess.PIPE, universal_newlines=True)
        
        # Obtain baseline AUC 
        lines = job.stdout.split('\n')
        
        for line in lines:
            if line.startswith("# AUC"):
                baseline_AUC = float(line.split()[2])
                
                if baseline_AUC < 0.5:
                    baseline_AUC = 1 - baseline_AUC
        
        ## RUN THROUGH ALL FEATURES ##
        for feature in X_train_impute:
            # Shuffle this feature in validation set and get new AUCs for model
            for i in range(n_runs_permutation_importance):
                # Shuffle feature
                X_validation_impute_this_run = copy.deepcopy(X_validation_impute)
                X_validation_impute_this_run[feature] = np.random.permutation(X_validation_impute[feature])
                
                # Define y-pred
                y_pred = clf.predict_proba(X_validation_impute_this_run)[:,1]
                
                # Combine outcome and predictor and write to file
                roc_df = pd.DataFrame()
                roc_df['Validation'] = y_validation_impute.reset_index(drop = True)
                roc_df['Prediction'] = y_pred
                
                # File name 
                file_name = str('/home/projects/ssi_10004/people/sofhor/roc_files/roc_file_permutation_importances_t' + str(t_set) + '_v' + str(v_set) + '_' + feature + str(i) + '.csv')
                
                # Write y pred and y validation to file
                roc_df.to_csv(file_name, sep = "\t", header = None, index = False)
                
                # Run roc on the resulting file
                job = subprocess.run(['./roc', '-pc', file_name], stdout=subprocess.PIPE, universal_newlines=True)
                
                # Obtain AUC 
                lines = job.stdout.split('\n')
                
                for line in lines:
                    if line.startswith("# AUC"):
                        AUC = float(line.split()[2])
                        
                        if AUC < 0.5:
                            AUC = 1 - AUC
                            
                # Find AUC difference from baseline
                AUC_diff = baseline_AUC - AUC
                
                # Add to dict
                permutation_importances_all_features[t_set][feature] += [AUC_diff]
                        

        

## Export AUCs
AUC_df = pd.DataFrame(columns=['Test_set', 'Feature', 'AUC'], index=range(number_of_sets*len(AUCs_all_features[0])))

row_number = 0

for test_set in AUCs_all_features:
    for feature in AUCs_all_features[test_set]:
        AUC = AUCs_all_features[test_set][feature]
        
        AUC_df.iloc[row_number,0] = test_set
        AUC_df.iloc[row_number,1] = feature
        AUC_df.iloc[row_number,2] = AUC
        
        row_number += 1

print("auc df")
print(AUC_df)

# Export dataframe
AUC_df.to_csv('/home/projects/ssi_10004/people/sofhor/AUCs_individual_features_whole_data.csv', index = True)


## Export RF importances  
RF_importances_df = pd.DataFrame(columns=['Test_set', 'Feature', 'RF_importance'], index=range(number_of_sets*len(feature_importances_all_features[0])))

row_number = 0

for test_set in feature_importances_all_features:
    for feature in feature_importances_all_features[test_set]:
        importance = feature_importances_all_features[test_set][feature]
        
        RF_importances_df.iloc[row_number,0] = test_set
        RF_importances_df.iloc[row_number,1] = feature
        RF_importances_df.iloc[row_number,2] = importance
        
        row_number += 1

print("RF importances df")
print(RF_importances_df)

# Export dataframe
RF_importances_df.to_csv('/home/projects/ssi_10004/people/sofhor/RF_feature_importances_whole_data.csv', index = True)


## Export permutation importances  
permutation_importances_df = pd.DataFrame(columns=['Test_set', 'Feature', 'Run', 'Permutation_Importance'], index=range(number_of_sets*len(permutation_importances_all_features[0])*len(permutation_importances_all_features[0]['age_delivery'])))

row_number = 0

for test_set in permutation_importances_all_features:
    for feature in permutation_importances_all_features[test_set]:
        importances = permutation_importances_all_features[test_set][feature]
        
        for i in range(len(importances)):
            permutation_importances_df.iloc[row_number,0] = test_set
            permutation_importances_df.iloc[row_number,1] = feature
            permutation_importances_df.iloc[row_number,2] = i
            permutation_importances_df.iloc[row_number,3] = importances[i]
            
            row_number += 1
        
        
print(permutation_importances_all_features)
print("Permutation importances df")
print(permutation_importances_df)

# Export dataframe
permutation_importances_df.to_csv('/home/projects/ssi_10004/people/sofhor/permutation_feature_importances_whole_data.csv', index = True)

    








