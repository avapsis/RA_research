import numpy as np
import csv
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from statistics import *


# DATA PROCESSING
#path = "/Users/annitavapsi/Dropbox (MIT)/MGH-All-Data/NSQIP_Emergency/Data/data_processed/Elective/"
path = "/home/gridsan/avapsi/data/reprocessed_data/Elective/"

train_X_mortal =  pd.read_csv(path+"train_sample_X_mort.csv")
train_y_mortal =  pd.read_csv(path+"train_sample_y_mort.csv").MORT
test_X_mortal = pd.read_csv(path+"test_sample_X_mort.csv")
test_y_mortal =  pd.read_csv(path+"test_sample_y_mort.csv").MORT

train_X_morbid =  pd.read_csv(path+"train_sample_X_morb.csv")
train_y_morbid =  pd.read_csv(path+"train_sample_y_morb.csv").MORB_ANY
test_X_morbid =  pd.read_csv(path+"test_sample_X_morb.csv")
test_y_morbid =  pd.read_csv(path+"test_sample_y_morb.csv").MORB_ANY;


indx_train_mort = train_X_mortal.SURGSPEC!= 7
indx_test_mort = test_X_mortal.SURGSPEC!=7
train_X_mortal = train_X_mortal[indx_train_mort]
train_y_mortal = train_y_mortal[indx_train_mort]
test_X_mortal = test_X_mortal[indx_test_mort]
test_y_mortal = test_y_mortal[indx_test_mort];

indx_train_morb = train_X_morbid.SURGSPEC!= 7
indx_test_morb = test_X_morbid.SURGSPEC!=7
train_X_morbid = train_X_morbid[indx_train_morb]
train_y_morbid = train_y_morbid[indx_train_morb]
test_X_morbid = test_X_morbid[indx_test_morb]
test_y_morbid = test_y_morbid[indx_test_morb];


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


train_X_mortal, test_X_mortal = one_hot_standard(train_X_mortal[X_columns],test_X_mortal[X_columns])
train_X_morbid, test_X_morbid = one_hot_standard(train_X_morbid[X_columns],test_X_morbid[X_columns])


# test different values of sampling_strategy 
# oversampling should be applied to just training set
# oversample the minority class to about 0.15 ratio
def logit_cv(model, X_train, y_train):
    #evaluate pipeline
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X_train, y_train, scoring='roc_auc', cv=cv, n_jobs=-1)
    print('Mean ROC AUC: %.3f' % mean(scores))
    

model = LogisticRegression(max_iter = 10000, verbose = 1) 


print("Mortality Run")
print("Started cross validation fit...") 
logit_cv(model, train_X_mortal, train_y_mortal)
print("Finished cross validation fit...")

print("Fitting final model...")
# logistic regression model - mortality change to .MORB for morbidity
model.fit(train_X_mortal,train_y_mortal)
y_pred = model.predict_proba(test_X_mortal)
logit_roc_auc_mortal = roc_auc_score(test_y_mortal, y_pred[:,1])
print("Out of sample logistic regression = ",logit_roc_auc_mortal)
print("Assessing model perfomrance on test set...")


print("Morbidity Run")
print("Started cross validation fit...") 
logit_cv(model, train_X_morbid, train_y_morbid)
print("Finished cross validation fit...")

print("Fitting final model...")
# logistic regression model - mortality change to .MORB for morbidity
model.fit(train_X_morbid,train_y_morbid)
y_pred = model.predict_proba(test_X_morbid)
logit_roc_auc_morbid = roc_auc_score(test_y_morbid, y_pred[:,1])
print("Out of sample logistic regression = ",logit_roc_auc_morbid)
print("Assessing model perfomrance on test set...")
