import lightgbm as lgb
import numpy as np
import csv
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from statistics import *

# ELECTIVE SURGERY DATA
# path = "/Users/annitavapsi/Dropbox (MIT)/MGH-All-Data/NSQIP_Emergency/Data/data_processed/Elective/"
path = "/home/gridsan/avapsi/data/reprocessed_data/Elective/"

print("Started reading data...")
X_train_mortal = pd.read_csv(path+"train_sample_X_mort.csv")
y_train_mortal = pd.read_csv(path+"train_sample_y_mort.csv").MORT
X_test_mortal = pd.read_csv(path+"test_sample_X_mort.csv")
y_test_mortal = pd.read_csv(path+"test_sample_y_mort.csv").MORT

X_train_morbid = pd.read_csv(path+"train_sample_X_morb.csv")
y_train_morbid = pd.read_csv(path+"train_sample_y_morb.csv").MORB_ANY
X_test_morbid = pd.read_csv(path+"test_sample_X_morb.csv")
y_test_morbid = pd.read_csv(path+"test_sample_y_morb.csv").MORB_ANY

print("Finished reading data...")


# columns to keep for analysis 
X_columns = [
    "SEX","RACE_NEW","ETHNICITY_HISPANIC", "INOUT","Age", "SURGSPEC","DIABETES","SMOKE",
    "DYSPNEA","FNSTATUS2", "VENTILAT","HXCOPD","ASCITES","HXCHF","HYPERMED","RENAFAIL","DIALYSIS","DISCANCR",
    "WNDINF","STEROID","WTLOSS","BLEEDDIS","TRANSFUS","PRSEPIS","PRSODM","PRBUN","PRCREAT","PRALBUM","PRBILI",
    "PRSGOT","PRALKPH","PRWBC","PRHCT","PRPLATE","PRPTT","PRINR","BMI"
]

# mortality
X_train_mortal = X_train_mortal[X_columns]
X_test_mortal = X_test_mortal[X_columns]

# morbidity
X_train_morbid = X_train_morbid[X_columns]
X_test_morbid = X_test_morbid[X_columns]


# ELECTIVE SURGERY: creating validation set
X_train_mortal, X_val_mortal = train_test_split(X_train_mortal, test_size=0.2, random_state = 403)
y_train_mortal, y_val_mortal = train_test_split(y_train_mortal, test_size=0.2, random_state = 403)

X_train_morbid, X_val_morbid = train_test_split(X_train_morbid, test_size=0.2, random_state = 403)
y_train_morbid, y_val_morbid = train_test_split(y_train_morbid, test_size=0.2, random_state = 403)



# ## EMERGENCY SURGERY - need for ICU: creating validation set
# X_train, X_val = train_test_split(X_train, test_size=0.2, random_state = 403)
# y_train, y_val = train_test_split(y_train, test_size=0.2, random_state = 403)


def categorical_features_specs(X_train, X_val, X_test):
    features_ordered_categorical = ["DYSPNEA", "PRSEPIS", "FNSTATUS2"]
    # dealing with odered categorical features - not sure if this is necessary for lgbm
    order_cat_DYSPNEA = CategoricalDtype(categories=[0, 1, 2], ordered=True)  
    order_cat_PRSEPIS = CategoricalDtype(categories=[0, 1, 2], ordered=True)
    order_cat_FNSTATUS2 = CategoricalDtype(categories=[0, 1, 2], ordered=True)
    for col_name in features_ordered_categorical:
        X_train.loc[:,col_name] = X_train.loc[:,col_name].astype(
            vars()[f"order_cat_{col_name}"]
        )
        X_val.loc[:,col_name] = X_val.loc[:,col_name].astype(
            vars()[f"order_cat_{col_name}"]
        )
        X_test.loc[:,col_name] = X_test.loc[:,col_name].astype(
            vars()[f"order_cat_{col_name}"]
        )
        print(f"Transformed column {col_name} to ordered categorical")

    # dealing with non-odered categorical features - not sure if this is necessary for lgbm ###########
    features_non_ordered_categorical = [
        "SEX","RACE_NEW","ETHNICITY_HISPANIC", "INOUT","SURGSPEC","DIABETES","SMOKE", 
        "VENTILAT","HXCOPD","ASCITES",
        "HXCHF","HYPERMED","RENAFAIL","DIALYSIS","DISCANCR","WNDINF","STEROID","WTLOSS",
        "BLEEDDIS","TRANSFUS"
    ]
    
    for col_name in features_non_ordered_categorical:
        X_train.loc[:,col_name] = X_train.loc[:,col_name].astype("category")
        X_val.loc[:,col_name] = X_val.loc[:,col_name].astype("category")
        X_test.loc[:,col_name] = X_test.loc[:,col_name].astype("category")
        print(f"Transformed column {col_name} to non_ordered categorical")
        
    return X_train, X_val, X_test


# ELECTIVE SURGERY: categorical feature specification
X_train_mortal, X_val_mortal, X_test_mortal = categorical_features_specs(X_train_mortal, X_val_mortal, X_test_mortal)
X_train_morbid, X_val_morbid, X_test_morbid = categorical_features_specs(X_train_morbid, X_val_morbid, X_test_morbid)

# # EMERGENCY SURGERY: categorical feature specification
# X_train, X_val, X_test = categorical_features_specs(X_train, X_val, X_test)


def boosting_train(X_train, X_val, y_train, y_val):
    print("-----Initiating Cross Validation for Parameter Tunning ------")
    # parameter tuning
    #Set the accuracy score = min(auc) = 0
    auc_max = 0
    count = 0 #Used for keeping track of the iteration number
    #How many runs to perform using randomly selected hyperparameters
    iterations = 50
    pp={}
    for i in range(iterations):
        print('iteration number', count)
        count += 1 #increment count
        try:
            d_train = lgb.Dataset(X_train, label=y_train) #Load in data
            params = {} #initialize parameters
            params['learning_rate'] = np.random.uniform(0, 1)
            params['objective'] = 'binary'
            params['metric'] = 'auc'
            params['num_leaves'] = np.random.randint(20, 300)
            params['min_data_in_leaf'] = np.random.randint(10, 100)
            params['max_depth'] = np.random.randint(5, 200) #this is too big
            iterations = np.random.randint(10, 10000)
            print(params, iterations)
            #Train using selected parameters
            clf = lgb.train(params, d_train, iterations)

            print("ok1")
            y_pred = clf.predict(X_val) #Create predictions on test set
            print(y_val)
            print(y_pred)
            print(y_pred.shape)
            print(y_val.shape)
            print("ok2")
            print(roc_auc_score(y_val,y_pred))
            auc=roc_auc_score(y_val,y_pred)
            print("ok3")
            print('AUC:', auc)
            if auc > auc_max:
                auc_max = auc
                pp = params 
        #in case something goes wrong
        except: 
            print('failed with')
            print(params)
            print("*" * 50)
    print('Maximum AUC achieved is: ', auc_max)
    print('Used params', pp)
    print("-----Finished Cross Validation for Parameter Tunning ------")
    
    return pp


params_mortal = boosting_train(X_train_mortal, X_val_mortal, y_train_mortal, y_val_mortal)
params_morbid = boosting_train(X_train_morbid, X_val_morbid, y_train_morbid, y_val_morbid)


def out_of_sample_auc(X_train, y_train, X_val, y_val, X_test, y_test, params):
    train = lgb.Dataset(X_train, label = y_train)
    val = lgb.Dataset(X_val, label = y_val)

    print("-----Initiating Fitting Test set on best params ------")
    num_round = 200
    bst = lgb.train(params, train, num_round, valid_sets=[val])
    y_pred = bst.predict(X_test, num_iteration=bst.best_iteration)

    auc_test = roc_auc_score(y_test,y_pred)

    print("-----Finished Fitting Test set on best params ------")
    print("Cross validated AUC:" , auc_test)
    return auc_test



auc_mortal = out_of_sample_auc(
    X_train_mortal, y_train_mortal, 
    X_val_mortal, y_val_mortal, 
    X_test_mortal, y_test_mortal, 
    params_mortal
)

auc_morbid = out_of_sample_auc(
    X_train_morbid, y_train_morbid, 
    X_val_morbid, y_val_morbid, 
    X_test_morbid, y_test_morbid, 
    params_morbid
)



