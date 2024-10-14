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
import random
import math

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
    
# Define list of all features
all_features = list(df.columns)

## Start features ##
# Exclude correlated features?
no_correlated_features_thank_you = True

# Number of features
n_features_start = 1

# Define number of steps to try for each test set
number_steps = 1000

# Possible actions in each step
actions = ["add", "remove", "swap"]

# Price per feature (because we want as simple a model as possible)
lambda_ = 0.00001

# Counter for total number of accepted steps
total_accepted_steps = 5 # Starts at 5 because the initial step of choosing start features will always be accepted and is not counted later

# Dict for storing AUCs of accepted steps
test_AUC_accepted_steps_dict = dict()
val_AUC_accepted_steps_dict = dict()

# List for storing AUC_diffs for k
AUC_diffs = []

# Split into five datasets that can take turn being test set
number_of_sets = 5
df_split = np.array_split(df, number_of_sets)
outcome_split = np.array_split(outcome_col, number_of_sets)
set_list = list(range(number_of_sets))

## Do everything separately for each test set ##
for t_set in range(number_of_sets):
    print("TEST SET:", t_set)
    
    # Define initial k and factor to scale k with
    k = 0.03228
    k_scaling = 0.881
    
    # Create subdicts for test sets
    test_AUC_accepted_steps_dict[t_set] = dict()
    val_AUC_accepted_steps_dict[t_set] = dict()
    
    # Define test set 
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
    
    # Define full training set to use for imputation and correlations check
    X_train_full = pd.concat(df_split_wo_test)
    y_train_full = pd.concat(outcome_split_wo_test)
    
    # Impute full training set and test set
    X_train_full_impute = imputation_method_train(X_train_full)
    
    if imputation_method == "zeros":
     #   X_test_impute = imputation_method_test(X_test)
        y_test_impute_full = imputation_method_test(y_test)
    else:
    #    X_test_impute = imputation_method_test(X_train_full, X_test)
        y_test_impute_full = imputation_method_test(y_train_full, y_test)
    
    # Define start list of features (with or without correlated features)
    if no_correlated_features_thank_you:
        # Calculate correlations
        correlations = X_train_full_impute.corr('spearman') # Morten har bedt om Spearman
        
        # Define features start list
        features_start = []
        
        while len(features_start) < n_features_start:
            new_feature = random.choice(np.setdiff1d(all_features, features_start))
            
            no_correlation_flag = True
            
            for feature in features_start:
                if correlations[feature][new_feature] > 0.7 or correlations[feature][new_feature] < -0.7:
                    no_correlation_flag = False
                    
            if no_correlation_flag:
                features_start += [new_feature]
            
    else:
        df_start = df.sample(n = n_features_start, axis = 'columns')
        features_start = list(df_start.columns)
        
    ### Run MC FIRST STEP ###
    # List for storing features for each accepted step
    chosen_features = []
    chosen_features += [features_start]
    
    # Run model for all test sets with the first random feature combination to get AUC0
    current_features = features_start
    
    ## Find AUCs of first feature selection
    # Create lists for storing validation y and prediction y for these features
    y_validation_list = []
    y_pred_val_list = []
    AUCs_test_set_predictions = []
    y_pred_test_list = []
        
    ## Run models for the currently chosen features ##
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
        
        # Impute training set, test set and validation set
        X_train_impute = imputation_method_train(X_train)
        y_train_impute = imputation_method_train(y_train)
        
        if imputation_method == "zeros":
            X_validation_impute = imputation_method_test(X_validation)
            y_validation_impute = imputation_method_test(y_validation)
            X_test_impute = imputation_method_test(X_test)
            y_test_impute = imputation_method_test(y_test)
        else:
            X_validation_impute = imputation_method_test(X_train, X_validation)
            y_validation_impute = imputation_method_test(y_train, y_validation)
            X_test_impute = imputation_method_test(X_train, X_test)
            y_test_impute = imputation_method_test(y_train, y_test)
        
        # Add validation y to list of validation ys for this test/train split
        y_validation_list.append(np.array(y_validation_impute.values))
        
        # Training, validation, and test data set with only the start features
        X_train_test_features = X_train_impute[current_features]
        X_validation_test_features = X_validation_impute[current_features]
        X_test_test_features = X_test_impute[current_features]
        
        # Run RF
        clf = RandomForestClassifier()
        clf.fit(X_train_test_features,y_train_impute)
        
        # Get prediction of RF with these chosen features on validation set
        y_pred = clf.predict_proba(X_validation_test_features)[:,1]
        
        # Add y_pred for this validation set to list of y_preds for this test set (with these features)
        y_pred_val_list += [y_pred]
        
        # y_pred for test set - with this training set combination (3/5 of data)
        y_pred_test_set = clf.predict_proba(X_test_test_features)[:,1]
        
        # Add y_pred_test for this training set combination to list of y_preds for this test set (with these features)
        y_pred_test_list += [y_pred_test_set]
   
    ## FIND AUC for validation sets (concatenated into one long y-pred)
    # Concatenate y-preds and y-validations and get AUC
    y_pred_this_test_and_feature_set = np.concatenate(y_pred_val_list)
    y_val_this_test_and_feature_set = np.concatenate(y_validation_list)
    
    # Print y pred and y val for this test set to file
    # Combine outcome and predictor and write to file
    roc_df = pd.DataFrame()
    roc_df['Validation'] = y_val_this_test_and_feature_set
    roc_df['Prediction'] = y_pred_this_test_and_feature_set
    
    # File name 
    file_name = str('/home/projects/ssi_10004/people/sofhor/roc_files/roc_file_VALIDATIONS_t' + str(t_set) + '_MC_start.csv')
    
    # Write y pred and y validation to file
    roc_df.to_csv(file_name, sep = "\t", header = None, index = False)
    
    # Run roc on the resulting file
    job = subprocess.run(['./roc', '-pc', file_name], stdout=subprocess.PIPE, universal_newlines=True)
    
    # Obtain AUC and store in dict
    lines = job.stdout.split('\n')
    
    for line in lines:
        if line.startswith("# AUC"):
            AUC = float(line.split()[2])
            
            if AUC < 0.5:
                AUC = 1 - AUC
    
    AUC_reduced = AUC - lambda_ 
    
    val_AUC_accepted_steps_dict[t_set]['0'] = dict()
    val_AUC_accepted_steps_dict[t_set]['0']['AUC'] = AUC
    val_AUC_accepted_steps_dict[t_set]['0']['AUC_reduced'] = AUC_reduced 
    val_AUC_accepted_steps_dict[t_set]['0']['Features'] = current_features.copy()
            
    ## FIND AUC for test set (ensemble model where prediction is the mean of predictions based on each training set combination (3/5))
    # Mean of predictions for test set
    y_pred_test_mean = np.mean(np.array(y_pred_test_list), axis = 0)
    
    # Print y pred and y val for this test set to file
    # Combine outcome and predictor and write to file
    roc_df = pd.DataFrame()
    roc_df['Validation'] = y_test_impute_full
    roc_df['Prediction'] = y_pred_test_mean
    
    # File name 
    file_name = str('/home/projects/ssi_10004/people/sofhor/roc_files/roc_file_TEST_t' + str(t_set) + 'MC_start.csv')
    
    # Write y pred and y validation to file
    roc_df.to_csv(file_name, sep = "\t", header = None, index = False)
    
    # Run roc on the resulting file
    job = subprocess.run(['./roc', '-pc', file_name], stdout=subprocess.PIPE, universal_newlines=True)
    
    # Obtain AUC and store in dict
    lines = job.stdout.split('\n')
    
    for line in lines:
        if line.startswith("# AUC"):
            AUC = float(line.split()[2])
            
            if AUC < 0.5:
                AUC = 1 - AUC
  
    test_AUC_accepted_steps_dict[t_set]['0'] = dict()
    test_AUC_accepted_steps_dict[t_set]['0']['AUC'] = AUC
    test_AUC_accepted_steps_dict[t_set]['0']['Features'] = current_features.copy()
    
    
    ### RUN MC STEPS ###
    for i in range(1,number_steps):
        print("step:", i)
        # Define action in this step
        action = random.choice(actions)
        
        print('action:', action)
        
        # Current features
        current_features = chosen_features[-1].copy()
        
        # Currently not chosen features
        not_current_features = np.setdiff1d(all_features, current_features)
        
        # Flag for running this model (false if no features were added or removed due to correlations)
        run_model = True
        
        ## Perform action ##
        if action == "add":
            new_feature = random.choice(not_current_features)
            #print("new feature:", new_feature)
            
            # Check correlations
            if no_correlated_features_thank_you:
                no_correlation_flag = True
                
                for feature in current_features:
                    if correlations[feature][new_feature] > 0.7 or correlations[feature][new_feature] < -0.7:
                        no_correlation_flag = False
                
                if no_correlation_flag:
                    current_features.append(new_feature)
                else:
                    run_model = False
            else:
                current_features.append(new_feature)
                
        elif action == "remove":
            # If there is only one feature to remove, this should not be done, and the model should not run
            if len(current_features) == 1:
                run_model = False
            else:
                remove_feature = random.choice(current_features)
                current_features.remove(remove_feature)

            
        elif action == "swap":
            new_feature = random.choice(not_current_features)
            #print("new feature:", new_feature)
            
            # Check correlations
            if no_correlated_features_thank_you:
                no_correlation_flag = True
                
                for feature in current_features:
                    if correlations[feature][new_feature] > 0.7 or correlations[feature][new_feature] < -0.7:
                        no_correlation_flag = False
                
                if no_correlation_flag:
                    remove_feature = random.choice(current_features)
                    current_features.remove(remove_feature)
                    current_features.append(new_feature)
                else:
                    run_model = False
            else:
                remove_feature = random.choice(current_features)
                current_features.remove(remove_feature)
                current_features.append(new_feature)
        
        ## Run model ##
        if run_model:
            ## Find AUCs of first feature selection
            # Create lists for storing validation y and prediction y for these features
            y_validation_list = []
            y_pred_val_list = []
            AUCs_test_set_predictions = []
            y_pred_test_list = []
                
            ## Run models for the currently chosen features ##
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
                
                # Impute training set, test set and validation set
                X_train_impute = imputation_method_train(X_train)
                y_train_impute = imputation_method_train(y_train)
                
                if imputation_method == "zeros":
                    X_validation_impute = imputation_method_test(X_validation)
                    y_validation_impute = imputation_method_test(y_validation)
                    X_test_impute = imputation_method_test(X_test)
                    y_test_impute = imputation_method_test(y_test)
                else:
                    X_validation_impute = imputation_method_test(X_train, X_validation)
                    y_validation_impute = imputation_method_test(y_train, y_validation)
                    X_test_impute = imputation_method_test(X_train, X_test)
                    y_test_impute = imputation_method_test(y_train, y_test)
                
                # Add validation y to list of validation ys for this test/train split
                y_validation_list.append(np.array(y_validation_impute.values))
                
                # Training, validation, and test data set with only the start features
                X_train_test_features = X_train_impute[current_features]
                X_validation_test_features = X_validation_impute[current_features]
                X_test_test_features = X_test_impute[current_features]
                
                # Run RF
                clf = RandomForestClassifier()
                clf.fit(X_train_test_features,y_train_impute)
                
                # Get prediction of RF with these chosen features on validation set
                y_pred = clf.predict_proba(X_validation_test_features)[:,1]
                
                # Add y_pred for this validation set to list of y_preds for this test set (with these features)
                y_pred_val_list += [y_pred]
                
                # y_pred for test set - with this training set combination (3/5 of data)
                y_pred_test_set = clf.predict_proba(X_test_test_features)[:,1]
                
                # Add y_pred_test for this training set combination to list of y_preds for this test set (with these features)
                y_pred_test_list += [y_pred_test_set]
                
           
            ## FIND AUC for validation sets (concatenated into one long y-pred)
            # Concatenate y-preds and y-validations and get AUC
            y_pred_this_test_and_feature_set = np.concatenate(y_pred_val_list)
            y_val_this_test_and_feature_set = np.concatenate(y_validation_list)
            
            # Print y pred and y val for this test set to file
            # Combine outcome and predictor and write to file
            roc_df = pd.DataFrame()
            roc_df['Validation'] = y_val_this_test_and_feature_set
            roc_df['Prediction'] = y_pred_this_test_and_feature_set
            
            # File name 
            file_name = str('/home/projects/ssi_10004/people/sofhor/roc_files/roc_file_VALIDATIONS_t' + str(t_set) + '_MC_' + str(i) +'.csv')
            
            # Write y pred and y validation to file
            roc_df.to_csv(file_name, sep = "\t", header = None, index = False)
            
            # Run roc on the resulting file
            job = subprocess.run(['./roc', '-pc', file_name], stdout=subprocess.PIPE, universal_newlines=True)
            
            # Obtain AUC and store in dict
            lines = job.stdout.split('\n')
            
            for line in lines:
                if line.startswith("# AUC"):
                    AUC = float(line.split()[2])
                    
                    if AUC < 0.5:
                        AUC = 1 - AUC
            
            # "Price" defined by number of features
            AUC_reduced = AUC - len(current_features)*lambda_
            
            # Get latest (reduced) AUC of accepted feature combinations from dict
            AUC0 = float(val_AUC_accepted_steps_dict[t_set][list(val_AUC_accepted_steps_dict[t_set])[-1]]['AUC_reduced'])
            
            print('AUC0:', AUC0)
            print('new AUC:', AUC_reduced)
            
            # Find difference in latest AUC and current AUC (with reductions based on number of features used for each model)
            AUC_diff = AUC_reduced - AUC0
            
            # Add to list for finding start k
            if AUC_diff < 0:
                AUC_diffs += [AUC_diff]
            
            # Check whether to accept the feature combination in this step or not #
            accept = False
            
            # Scale k
            if i%10 == 0:
                k = k*k_scaling
            if k <= 0:
                k = 0.0000001
            print("k this round:", k)
            
            if AUC_diff > 0:
                accept = True
            elif AUC_diff < 0:    
                # Define p
                p = math.exp(AUC_diff/k)
                
                # Draw random number to determine whether to accept with this p or not
                random_number = random.random()
                
                # Choose to accept or not based on p and random number
                if p > random_number:
                    accept = True
            
            print('AUC_diff:', AUC_diff)
            if AUC_diff < 0:
                print('p:', p)
                print('random number:', random_number)
            print('accept?', accept)
            
            # Save AUC and feature combination if the feature combination was accepted
            if accept:
                # Save AUC for validation sets
                val_AUC_accepted_steps_dict[t_set][str(i)] = dict()
                val_AUC_accepted_steps_dict[t_set][str(i)]['AUC'] = AUC
                val_AUC_accepted_steps_dict[t_set][str(i)]['AUC_reduced'] = AUC_reduced
                val_AUC_accepted_steps_dict[t_set][str(i)]['Features'] = current_features.copy() 
                
                chosen_features += [current_features.copy()]
                
                ## FIND AUC for test set if this step was accepted (ensemble model where prediction is the mean of predictions based on each training set combination (3/5))
                # Mean of predictions for test set
                y_pred_test_mean = np.mean(np.array(y_pred_test_list), axis = 0)
                
                # Print y pred and y val for this test set to file
                # Combine outcome and predictor and write to file
                roc_df = pd.DataFrame()
                roc_df['Validation'] = y_test_impute_full
                roc_df['Prediction'] = y_pred_test_mean
                
                # File name 
                file_name = str('/home/projects/ssi_10004/people/sofhor/roc_files/roc_file_TEST_t' + str(t_set) + 'MC_' + str(i) + '.csv')
                
                # Write y pred and y validation to file
                roc_df.to_csv(file_name, sep = "\t", header = None, index = False)
                
                # Run roc on the resulting file
                job = subprocess.run(['./roc', '-pc', file_name], stdout=subprocess.PIPE, universal_newlines=True)
                
                # Obtain AUC and store in dict
                lines = job.stdout.split('\n')
                
                for line in lines:
                    if line.startswith("# AUC"):
                        AUC = float(line.split()[2])
                        
                        if AUC < 0.5:
                            AUC = 1 - AUC
                
                test_AUC_accepted_steps_dict[t_set][str(i)] = dict()
                test_AUC_accepted_steps_dict[t_set][str(i)]['AUC'] = AUC
                test_AUC_accepted_steps_dict[t_set][str(i)]['Features'] = current_features.copy()
                
                # Count total number of accepted steps for all test sets
                total_accepted_steps += 1
        


# Export AUC dicts
file = open('/home/projects/ssi_10004/people/sofhor/val_AUC_feature_selection_MC.txt','w')
file.write(str(val_AUC_accepted_steps_dict))
file.close()

file = open('/home/projects/ssi_10004/people/sofhor/test_AUC_feature_selection_MC.txt','w')
file.write(str(test_AUC_accepted_steps_dict))
file.close()


# Create dataframe to save and use for plots - validation sets
val_AUC_df = pd.DataFrame(columns=['Test_set', 'Accepted_step', 'AUC', 'Number_of_features', 'Feature_list'], index=range(total_accepted_steps))

row_number = 0

for test_set in val_AUC_accepted_steps_dict:
    for accepted_step in val_AUC_accepted_steps_dict[test_set]:        
        val_AUC_df.iloc[row_number,0] = test_set
        val_AUC_df.iloc[row_number,1] = accepted_step
        val_AUC_df.iloc[row_number,2] = val_AUC_accepted_steps_dict[test_set][accepted_step]['AUC']
        val_AUC_df.iloc[row_number,3] = len(val_AUC_accepted_steps_dict[test_set][accepted_step]['Features'])
        val_AUC_df.iloc[row_number,4] = val_AUC_accepted_steps_dict[test_set][accepted_step]['Features']
    
        row_number += 1

# Export dataframe
val_AUC_df.to_csv('/home/projects/ssi_10004/people/sofhor/val_AUC_feature_selection_MC.csv', index = True)


# Create dataframe to save and use for plots - test sets
test_AUC_df = pd.DataFrame(columns=['Test_set', 'Accepted_step', 'AUC', 'Number_of_features', 'Feature_list'], index=range(total_accepted_steps))

row_number = 0

for test_set in test_AUC_accepted_steps_dict:
    for accepted_step in test_AUC_accepted_steps_dict[test_set]:        
        test_AUC_df.iloc[row_number,0] = test_set
        test_AUC_df.iloc[row_number,1] = accepted_step
        test_AUC_df.iloc[row_number,2] = test_AUC_accepted_steps_dict[test_set][accepted_step]['AUC']
        test_AUC_df.iloc[row_number,3] = len(test_AUC_accepted_steps_dict[test_set][accepted_step]['Features'])
        test_AUC_df.iloc[row_number,4] = test_AUC_accepted_steps_dict[test_set][accepted_step]['Features']
    
        row_number += 1

# Export dataframe
test_AUC_df.to_csv('/home/projects/ssi_10004/people/sofhor/test_AUC_feature_selection_MC.csv', index = True)


## Export chosen features 
# TEST SET 0 #
feature_list_from_dict_0 = val_AUC_accepted_steps_dict[0][list(val_AUC_accepted_steps_dict[0])[-1]]['Features']
features_test0 = pd.DataFrame(columns = ['Features'], index = range(len(feature_list_from_dict_0)))

row_number = 0

for feature in feature_list_from_dict_0:
    features_test0.iloc[row_number,0] = feature
    row_number += 1

features_test0.to_csv('/home/projects/ssi_10004/people/sofhor/val_features_MC_test0.csv', index = True)

# TEST SET 1 #
feature_list_from_dict_1 = val_AUC_accepted_steps_dict[1][list(val_AUC_accepted_steps_dict[1])[-1]]['Features']
features_test1 = pd.DataFrame(columns = ['Features'], index = range(len(feature_list_from_dict_1)))

row_number = 0

for feature in feature_list_from_dict_1:
    features_test1.iloc[row_number,0] = feature
    row_number += 1

features_test1.to_csv('/home/projects/ssi_10004/people/sofhor/val_features_MC_test1.csv', index = True)


# TEST SET 2 #
feature_list_from_dict_2 = val_AUC_accepted_steps_dict[2][list(val_AUC_accepted_steps_dict[2])[-1]]['Features']
features_test2 = pd.DataFrame(columns = ['Features'], index = range(len(feature_list_from_dict_2)))

row_number = 0

for feature in feature_list_from_dict_2:
    features_test2.iloc[row_number,0] = feature
    row_number += 1

features_test2.to_csv('/home/projects/ssi_10004/people/sofhor/val_features_MC_test2.csv', index = True)


# TEST SET 3 #
feature_list_from_dict_3 = val_AUC_accepted_steps_dict[3][list(val_AUC_accepted_steps_dict[3])[-1]]['Features']
features_test3 = pd.DataFrame(columns = ['Features'], index = range(len(feature_list_from_dict_3)))

row_number = 0

for feature in feature_list_from_dict_3:
    features_test3.iloc[row_number,0] = feature
    row_number += 1

features_test3.to_csv('/home/projects/ssi_10004/people/sofhor/val_features_MC_test3.csv', index = True)


# TEST SET 4 #
feature_list_from_dict_4 = val_AUC_accepted_steps_dict[4][list(val_AUC_accepted_steps_dict[4])[-1]]['Features']
features_test4 = pd.DataFrame(columns = ['Features'], index = range(len(feature_list_from_dict_4)))

row_number = 0

for feature in feature_list_from_dict_4:
    features_test4.iloc[row_number,0] = feature
    row_number += 1

features_test4.to_csv('/home/projects/ssi_10004/people/sofhor/val_features_MC_test4.csv', index = True)



## k start ##
print('')
print("k:", np.mean(AUC_diffs))














