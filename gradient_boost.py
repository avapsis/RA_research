
import lightgbm as lgb
import numpy as np
import csv
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from statistics import *

# gradient boosting for classification in scikit-learn
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from matplotlib import pyplot

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer

from sklearn.preprocessing import StandardScaler



# ELECTIVE SURGERY DATA
#path = "/Users/annitavapsi/Dropbox (MIT)/MGH-All-Data/NSQIP_Emergency/Data/data_processed/Elective/"
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


# removing SURGSPEC = 7 since it is very small - not representative 
indx_train_mort = X_train_mortal.SURGSPEC!= 7
indx_test_mort = X_test_mortal.SURGSPEC!=7
X_train_mortal = X_train_mortal[indx_train_mort]
y_train_mortal = y_train_mortal[indx_train_mort]
X_test_mortal = X_test_mortal[indx_test_mort]
y_test_mortal = y_test_mortal[indx_test_mort];

indx_train_morb = X_train_morbid.SURGSPEC!= 7
indx_test_morb = X_test_morbid.SURGSPEC!=7
X_train_morbid = X_train_morbid[indx_train_morb]
y_train_morbid = y_train_morbid[indx_train_morb]
X_test_morbid = X_test_morbid[indx_test_morb]
y_test_morbid = y_test_morbid[indx_test_morb];


# columns to keep for analysis 
X_columns = [
    "SEX","RACE_NEW","ETHNICITY_HISPANIC", "INOUT","Age","SURGSPEC","DIABETES","SMOKE",
    "DYSPNEA","FNSTATUS2", "VENTILAT","HXCOPD","ASCITES","HXCHF","HYPERMED","RENAFAIL","DIALYSIS","DISCANCR",
    "WNDINF","STEROID","WTLOSS","BLEEDDIS","TRANSFUS","PRSEPIS","PRSODM","PRBUN","PRCREAT","PRALBUM","PRBILI",
    "PRSGOT","PRALKPH","PRWBC","PRHCT","PRPLATE","PRPTT","PRINR","BMI"
]



def one_hot_standard(X_train, X_test):
    # one-hot-encode
    names = [
        "RACE_NEW", "DIABETES", "DYSPNEA", "SURGSPEC", "FNSTATUS2", "PRSEPIS" 
    ]
    
    X_train = pd.get_dummies(X_train, columns=names, prefix=names, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=names, prefix=names, drop_first=True)
    
    # STANDARDIZE # ONLY NON-CATEGORICAL?
    indices_to_scale = X_train.dtypes!=np.uint8
    scaler1 = StandardScaler(copy=False).fit(X_train.loc[:,indices_to_scale.values])
    
    # DATA scale
    X_train.loc[:,indices_to_scale.values] = scaler1.transform(X_train.loc[:,indices_to_scale.values])
    X_test.loc[:,indices_to_scale.values] = scaler1.transform(X_test.loc[:,indices_to_scale.values])
   
    return(X_train, X_test)

print(X_train_mortal.SURGSPEC.unique())
print(X_test_mortal.SURGSPEC.unique())

X_train_mortal, X_test_mortal = one_hot_standard(X_train_mortal[X_columns],X_test_mortal[X_columns])
X_train_morbid, X_test_morbid = one_hot_standard(X_train_morbid[X_columns],X_test_morbid[X_columns])

print("All ok with data processing")


def param_tune(X, y, params):
    model = GradientBoostingClassifier()
    clf = GridSearchCV(model, params ,scoring="roc_auc",refit=False,cv=2)
    clf.fit(X, y)
    #optimised_gb = clf.best_estimator_
    #print("Best estimator:", optimised_gb)
    return clf


params = {
    "loss":["deviance"],
    "learning_rate": [0.01, 0.1, 0.2],
    "min_samples_split": [0.1, 0.5],  #np.linspace(0.1, 0.5, 12),
    "min_samples_leaf": [0.1,0.5],  #np.linspace(0.1, 0.5, 12),
    #"max_depth":[3, 5, 8],
    "max_features":["log2","sqrt"],
    "criterion": ["friedman_mse"], #"mae"],
    "subsample":[0.5, 0.75, 1.0],
    #"n_estimators":[10]
    }

clf1 = param_tune(X_train_mortal,y_train_mortal,params=params)
print("clf_treat = ", clf1)
print("best_params_",clf1.best_params_)
print("best_score_",clf1.best_score_)


clf2 = param_tune(X_train_morbid,y_train_morbid,params=params)
print("clf_treat = ", clf2)
print("best_params_",clf2.best_params_)
print("best_score_",clf2.best_score_)
