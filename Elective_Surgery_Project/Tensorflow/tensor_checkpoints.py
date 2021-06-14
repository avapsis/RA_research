import numpy as np
import csv
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold


from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping 
import tensorflow as tf
import keras

from keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
import h5py

#path = "/Users/annitavapsi/Dropbox (MIT)/mgh_data_backup/Data/Elective/"
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

X_columns = ['SEX', 'RACE_NEW', 'ETHNICITY_HISPANIC', 'INOUT',
       'Age', 'SURGSPEC', 'DIABETES', 'SMOKE',
       'DYSPNEA', 'FNSTATUS2', 'VENTILAT', 'HXCOPD', 'ASCITES', 'HXCHF',
       'HYPERMED', 'RENAFAIL', 'DIALYSIS', 'DISCANCR', 'WNDINF', 'STEROID',
       'WTLOSS', 'BLEEDDIS', 'TRANSFUS', 'PRSEPIS', 'PRSODM', 'PRBUN',
       'PRCREAT', 'PRALBUM', 'PRBILI', 'PRSGOT', 'PRALKPH', 'PRWBC', 'PRHCT',
       'PRPLATE', 'PRPTT', 'PRINR', 'BMI']



X_train_mortal = X_train_mortal[X_columns]
X_test_mortal = X_test_mortal[X_columns]
X_train_morbid = X_train_morbid[X_columns]
X_test_morbid = X_test_morbid[X_columns]

print("All ok with data processing")

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


X_train_mortal, X_test_mortal = one_hot_standard(X_train_mortal,X_test_mortal)
X_train_morbid, X_test_morbid = one_hot_standard(X_train_morbid,X_test_morbid)


# takes list of numbers as arguments and depending on list length 
# builds a FNN of the same depth and numbers as hidden features
def model_params(lst, num_features, X_train, y_train, X_val, y_val, X_test, y_test, batch_size):
    tf.keras.backend.clear_session()
    length1 = len(lst)

    # define keras model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(lst[0], input_dim=num_features, activation= 'relu'))
    for i in range(length1)[1:-1]:
        model.add(tf.keras.layers.Dense(lst[i], activation='relu'))  
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    # compile keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC'])
    # early stopping
    es = EarlyStopping(monitor='val_auc', mode='max', verbose=1)
    mc = ModelCheckpoint('best_model.h5', monitor='val_auc', mode='max', verbose=1, save_best_only=True)
    # fit keras model
    model.fit(X_train, y_train, epochs=25, batch_size=batch_size, validation_data = (X_val, y_val), callbacks=[es,mc])
    # load best saved model
    saved_model = tf.keras.models.load_model('best_model.h5')
    
    _, auc_train = saved_model.evaluate(X_train, y_train)
    print('AUC_train: %.2f' % (auc_train*100))
    
    _, auc_val = saved_model.evaluate(X_val, y_val)
    print('AUC_val: %.2f' % (auc_val*100))
    
    _, auc_test = saved_model.evaluate(X_test, y_test)
    print('AUC_test: %.2f' % (auc_test*100))
    
    return(auc_val) #changing from test auc to val auc -- to make a choice between different parameters 
  

    
def cv_tensor(X,y,X_test,y_test,num_folds,lst, batch_list):
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=num_folds, shuffle=True)
    
    # 5-fold cross validation, model evaluation
    fold_no = 1
    results = []
    params_results = []
    for batch_size in batch_list:
        for fold_no, (train, val) in enumerate(kfold.split(X)):  # provides indices to split on
            # Generate a print
            print('------------------------------------------------------------------------')
            print(f'Training for fold {fold_no} ...')
            X_train = X.iloc[train]
            y_train = y.iloc[train]
            X_val = X.iloc[val]
            y_val = y.iloc[val]
            
            num_features = np.shape(X_train)[1]
            auc_val = model_params(lst, num_features, X_train, y_train, X_val, y_val, X_test, y_test, batch_size) #changing auc_test to auc_val
            results.append(auc_val) #changing auc_test to auc_val
            
        params_results.append([repr(lst),len(lst),batch_size,np.mean(results)])
        
        
    return(params_results)







print("Starting Training for Mortality...")
batch_list1 = [10000] #[10000,5000,2000,1000,100,50] #chance number of folds
all_results = [cv_tensor(X_train_mortal, y_train_mortal, X_test_mortal, y_test_mortal,2,lst=lst,batch_list = batch_list1) for lst in [
    #[1000, 950, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200, 250, 100, 60],
    #[1000, 900, 800, 700, 600, 500, 400, 300, 200, 100, 60],
    [1000, 800, 600, 400, 200, 60],
    #[1000, 500, 1000, 500, 60],
    #[1000, 500, 100, 60],
    #[800, 800],
    #[500, 500, 500],
    #[200, 200, 200, 200, 200],
    #[100, 100, 100, 100, 100],
    #[100, 90, 80, 70, 60],
    #[100, 60, 60, 60, 60],
    #[100, 60, 60, 60],
    #[100, 60, 60],
    #[100, 60],
    #[60]
    ]]
print("Finished Training for Mortality..")


for i in range(np.shape(all_results)[0]):
        for j in range(np.shape(all_results)[1]):
            if i==0 and j==0:
                df1 = pd.DataFrame([all_results[i][j]], columns=['nodes', 'depth', 'batch_size', 'performance'])
            else:
                df2 = pd.DataFrame([all_results[i][j]], columns=['nodes', 'depth', 'batch_size', 'performance'])
                df1 = df1.append(df2, ignore_index=True)

                
print("Starting Training for Morbidity...")
batch_list1 = [5000] #[10000,5000,2000,1000,100,50]
all_results = [cv_tensor(X_train_morbid, y_train_morbid, X_test_morbid, y_test_morbid,5,lst=lst,batch_list = batch_list1) for lst in [
    [1000, 950, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200, 250, 100, 60],
    #[1000, 900, 800, 700, 600, 500, 400, 300, 200, 100, 60],
    #[1000, 800, 600, 400, 200, 60],
    #[1000, 500, 1000, 500, 60],
    #[1000, 500, 100, 60],
    #[800, 800],
    #[500, 500, 500],
    #[200, 200, 200, 200, 200],
    #[100, 100, 100, 100, 100],
    #[100, 90, 80, 70, 60],
    #[100, 60, 60, 60, 60],
    #[100, 60, 60, 60],
    #[100, 60, 60],
    #[100, 60],
    #[60]
    ]]
print("Finished Training for Morbidity..")

for i in range(np.shape(all_results)[0]):
        for j in range(np.shape(all_results)[1]):
            if i==0 and j==0:
                df3 = pd.DataFrame([all_results[i][j]], columns=['nodes', 'depth', 'batch_size', 'performance'])
            else:
                df4 = pd.DataFrame([all_results[i][j]], columns=['nodes', 'depth', 'batch_size', 'performance'])
                df3 = df3.append(df4, ignore_index=True)


print("------------------------- Results ---------------------------")
print("MORTALITY TABLE")
print(df1)
df1.to_csv("/home/gridsan/avapsi/src_code/Tensor/new_data/outputs/results_mortal_31_05.csv", index=False) 



print("MORBIDITY TABLE")
print(df3)
df3.to_csv("/home/gridsan/avapsi/src_code/Tensor/new_data/outputs/results_morbid_31_05.csv", index=False) 


print("------------------------- Printing -------------------------")


print(df1)
print(df3)

