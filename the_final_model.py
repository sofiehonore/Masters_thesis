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
    
# Import top 30 features for each test set
top30_features = pd.read_csv('/home/projects/ssi_10004/people/sofhor/chosen_features_validation_sets_feature_selection_CV.csv', sep = ",")

# Import overlapping features for all test sets top 30s
overlapping_top30_features = pd.read_csv('/home/projects/ssi_10004/people/sofhor/feature_selection_redundancy_overlapping_top30_features.csv', sep = ",")

# Import overlapping features for all test sets top x
overlapping_x_features = pd.read_csv('/home/projects/ssi_10004/people/sofhor/feature_selection_redundancy_overlapping_x_features.csv', sep = ",")

# x for number of features to use for individual test sets
number_of_features_to_use = 18

## CV ##
# Split into five data sets that can take turns being test and train/validation
number_of_sets = 5
df_split = np.array_split(df, number_of_sets)
outcome_split = np.array_split(outcome_col, number_of_sets)

set_list = list(range(number_of_sets))

# Dict for storing AUCs
val_AUC_dict = dict()
test_AUC_dict = dict()
one_test_AUC_dict = dict()

# Lists for storing y's to find one AUC for the whole data set
y_pred_test_whole_data = []
y_real_test_whole_data = []

for method in ['individual_test_set_features', 'overlapping_top30', 'overlapping_topx']:
    print('Method:', method)
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
        
        # Subdict for storing AUCs of features for the training set of this test set + AUC of models based on these features and both validation and test set
        val_AUC_dict[t_set] = dict()
        test_AUC_dict[t_set] = dict()
        
        # Define full training set (everything except test set)
        y_train_full = pd.concat(outcome_split_wo_test)
        
        # Impute test set based on full training set
        if imputation_method == "zeros":
            y_test_impute_full = imputation_method_test(y_test)
        else:
            y_test_impute_full = imputation_method_test(y_train_full, y_test)
            
        # List for storing validation and pred ys for this test set
        y_validation_list = list()
        y_pred_list = list()
        y_pred_test_list = list()
        
        # Define features to use for this test set
        if method == 'individual_test_set_features':
            # Define features to be used in this test set 
            if t_set == 0:
                features_this_test_set = top30_features['test0'].tolist()[0:number_of_features_to_use]
            elif t_set == 1:
                features_this_test_set = top30_features['test1'].tolist()[0:number_of_features_to_use]
            elif t_set == 2:
                features_this_test_set = top30_features['test2'].tolist()[0:number_of_features_to_use]
            elif t_set == 3:
                features_this_test_set = top30_features['test3'].tolist()[0:number_of_features_to_use]
            elif t_set == 4:
                features_this_test_set = top30_features['test4'].tolist()[0:number_of_features_to_use]
        
        elif method == 'overlapping_top30':
            features_this_test_set = overlapping_top30_features['Features'].tolist()
        
        elif method == "overlapping_topx":
            features_this_test_set = overlapping_x_features['Features'].tolist()
            
        print("Using these features:")
        print(features_this_test_set)
            
        for v_set in range(len(remaining_sets)):
            print('validation_set:', v_set)
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
                X_test_impute = imputation_method_test(X_test)
                y_test_impute = imputation_method_test(y_test)
            else:
                X_validation_impute = imputation_method_test(X_train, X_validation)
                y_validation_impute = imputation_method_test(y_train, y_validation)
                X_test_impute = imputation_method_test(X_train, X_test)
                y_test_impute = imputation_method_test(y_train, y_test)
            
            ## Define training and validation set Xs with only subset of features that were overlapping in AUC/feature importances
            X_train_impute = X_train_impute[features_this_test_set]
            X_validation_impute = X_validation_impute[features_this_test_set]
            X_test_impute = X_test_impute[features_this_test_set]
            
            # Add validation y to list of validation ys for this test/train split
            y_validation_list.append(np.array(y_validation_impute.values))
            
            # Run RF
            clf = RandomForestClassifier()
            clf.fit(X_train_impute,y_train_impute)
            
            # Get prediction of RF with these chosen features
            y_pred = clf.predict_proba(X_validation_impute)[:,1]
            
            # Add y_pred for this validation set to list of y_preds for this test set (with these features)
            y_pred_list += [y_pred]
            
            ## FIND AUCs for test set
            # y_pred for test set - with this training set combination (3/5 of data)
            y_pred_test_set = clf.predict_proba(X_test_impute)[:,1]
            
            y_pred_test_list += [y_pred_test_set]
                
        ## FIND AUCs for validation sets
        # Concatenate y-preds and y-validations and get AUC
        y_pred_this_test_and_feature_set = np.concatenate(y_pred_list)
        y_val_this_test_and_feature_set = np.concatenate(y_validation_list)
        
        # Print y pred and y val for this test set to file
        # Combine outcome and predictor and write to file
        roc_df = pd.DataFrame()
        roc_df['Validation'] = y_val_this_test_and_feature_set
        roc_df['Prediction'] = y_pred_this_test_and_feature_set
        
        # File name 
        file_name = str('/home/projects/ssi_10004/people/sofhor/roc_files/roc_file_VALIDATIONS_t' + str(t_set) + '_final_model_' + method + '.csv')
        
        # Write y pred and y validation to file
        roc_df.to_csv(file_name, sep = "\t", header = None, index = False)
        
        # Run roc on the resulting file
        job = subprocess.run(['./roc', '-pc', file_name], stdout=subprocess.PIPE, universal_newlines=True)
        
        # Obtain AUC and store in dict
        lines = job.stdout.split('\n')
        
        for line in lines:
            if line.startswith("# AUC"):
                AUC = line.split()[2]
                
                if float(AUC) < 0.5:
                    AUC = 1 - float(AUC)
                
        val_AUC_dict[t_set] = float(AUC)
        
        ## FIND AUC for test set (ensemble model where prediction is the mean of predictions based on each training set combination (3/5))
        # Mean of predictions for test set
        y_pred_test_mean = np.mean(np.array(y_pred_test_list), axis = 0)
        
        # Add to big y-pred list for test predictions
        y_pred_test_whole_data += [y_pred_test_mean]
        y_real_test_whole_data += [y_test_impute_full]
        
        # Print y pred and y val for this test set to file
        # Combine outcome and predictor and write to file
        roc_df = pd.DataFrame()
        roc_df['Validation'] = y_test_impute_full
        roc_df['Prediction'] = y_pred_test_mean
        
        # File name 
        file_name = str('/home/projects/ssi_10004/people/sofhor/roc_files/roc_file_TEST_SET_t' + str(t_set) + '_final_model_' + method + '.csv')
        
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
                    
        test_AUC_dict[t_set] = AUC
        
    
    ### FIND ONE TEST AUC ###
    # based on the mean predictions for all test sets
    # Concatenate y-preds and y-validations and get AUC
    y_pred_whole_data = np.concatenate(y_pred_test_whole_data)
    y_test_whole_data = np.concatenate(y_real_test_whole_data)
    
    # Print y pred and y val for this test set to file
    # Combine outcome and predictor and write to file
    roc_df = pd.DataFrame()
    roc_df['Validation'] = y_test_whole_data
    roc_df['Prediction'] = y_pred_whole_data
    
    # File name 
    file_name = str('/home/projects/ssi_10004/people/sofhor/roc_files/roc_file_whole_test_y_pred_final_model_' + method + '.csv')
    
    # Write y pred and y validation to file
    roc_df.to_csv(file_name, sep = "\t", header = None, index = False)
    
    # Run roc on the resulting file
    job = subprocess.run(['./roc', '-pc', file_name], stdout=subprocess.PIPE, universal_newlines=True)
    
    # Obtain AUC and roc-curve coordinates and store in dict
    lines = job.stdout.split('\n')
    
    plot_flag = False
    AUC_flag = False
    
    one_test_AUC_dict[method] = dict()
    one_test_AUC_dict[method]['x_coord'] = list()
    one_test_AUC_dict[method]['y_coord'] = list()
    
    for line in lines:
        if line.startswith("# AUC"):
            AUC_flag = True
            AUC = np.float(line.split()[2])
            
            if AUC < 0.5:
                AUC = 1 - AUC
            
            one_test_AUC_dict[method]['AUC'] = AUC
            
            #print(one_test_AUC_dict)
        
        if line == " 0.00000  0.00000":
            plot_flag = True
        
        if plot_flag == True and AUC_flag == False:
            x_coord = float(line.split()[0])
            y_coord = float(line.split()[1])
            one_test_AUC_dict[method]['x_coord'] += [x_coord]
            one_test_AUC_dict[method]['y_coord'] += [y_coord]
            
    # Obtain AUC0.1
    # Run roc on the resulting file
    job = subprocess.run(['./roc', '-pc', '-f', '0.1', file_name], stdout=subprocess.PIPE, universal_newlines=True)
    
    # Obtain AUC and roc-curve coordinates and store in dict
    lines = job.stdout.split('\n')

    # Print AUCs to csv for plotting
    AUC_df = pd.DataFrame(columns=['Test_set', 'Test_sets', 'Validation_sets'], index=range(number_of_sets))
    
    for i in range(number_of_sets):
        AUC_df.iloc[i,0] = i
        AUC_df.iloc[i,1] = test_AUC_dict[i]
        AUC_df.iloc[i,2] = val_AUC_dict[i]

    
    # Export dataframe as csv
    AUC_df.to_csv(str('/home/projects/ssi_10004/people/sofhor/AUCs_final_model_' + method + '.csv'), index = True)


## Export ROC curve coordinates ##
# Individual features for all test sets 
roc_df_individual_features = pd.DataFrame(columns=['x_coord', 'y_coord'], index=range(len(one_test_AUC_dict['individual_test_set_features']['x_coord'])))
roc_df_individual_features['x_coord'] = one_test_AUC_dict['individual_test_set_features']['x_coord']
roc_df_individual_features['y_coord'] = one_test_AUC_dict['individual_test_set_features']['y_coord']

roc_df_individual_features.to_csv('/home/projects/ssi_10004/people/sofhor/roc_final_model_individual_features.csv', index = True)

# Same features for all test sets, based on overlap between top 30 features
roc_df_top30_features = pd.DataFrame(columns=['x_coord', 'y_coord'], index=range(len(one_test_AUC_dict['overlapping_top30']['x_coord'])))
roc_df_top30_features['x_coord'] = one_test_AUC_dict['overlapping_top30']['x_coord']
roc_df_top30_features['y_coord'] = one_test_AUC_dict['overlapping_top30']['y_coord']

roc_df_top30_features.to_csv('/home/projects/ssi_10004/people/sofhor/roc_final_model_overlap_top30_features.csv', index = True)

# Same features for all test sets, based on overlap between top x features
roc_df_topx_features = pd.DataFrame(columns=['x_coord', 'y_coord'], index=range(len(one_test_AUC_dict['overlapping_topx']['x_coord'])))
roc_df_topx_features['x_coord'] = one_test_AUC_dict['overlapping_topx']['x_coord']
roc_df_topx_features['y_coord'] = one_test_AUC_dict['overlapping_topx']['y_coord']
  
roc_df_individual_features.to_csv('/home/projects/ssi_10004/people/sofhor/roc_final_model_overlap_topx_features.csv', index = True)














