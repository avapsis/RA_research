# RA_research

This repo includes all files created as part of a research collaboration between MIT and MGH. 

Project 1: Predicting probability of morbidity and mortality of Elective Surgery patients - Elective_Surgery_Project.

Project 2: Predicting the probability that an Emergency surgery patient will need ICU admission post-operatively - Emergency_Surgery_Project . 

Several methods have been used including: Optimal Classification Trees, Gradient Boosting, TensorFlow and Logistic Regression.

#### Data
The data used for each of the projects are available in the MGH-All-Data Dropbox folder (NSQIP Dataset)

Elective Surgery Project data location: MGH-All-Data/NSQIP/NSQIP_Data_Processed/random_split_NSQIP/ElecSurg_data

Emergency Surgery Project data location: MGH-All-Data/NSQIP/NSQIP_Data_Processed/random_split_NSQIP/EmergSurg_data

#### Results 
Table of results (AUC) for each project are available at: MGH-All-Data/NSQIP/NSQIP_Results_Summary_15_05_2021.docx



#### Emergency Surgery project what you need to do to get the application:

The outputs of the OCT script are:
* OCT tree: MGH-All-Data/NSQIP/src_code/src_code_annita/OCT/oct_final_scripts/oct_emergency/results/seed=1___outcome=need_for_ICU___minbucket=100___nsqip_auc876.html
* json file used to create application: MGH-All-Data/NSQIP/src_code/src_code_annita/OCT/oct_final_scripts/oct_emergency/results/seed=1___outcome=need_for_ICU___minbucket=100___nsqip_auc876.json

results folder: MGH-All-Data/NSQIP/src_code/src_code_annita/OCT/oct_final_scripts/oct_emergency/results

To get the application you need to read it though the categorical_feature_mapping.ipynb file which decodes all categorical features (represented as numeric in the script run) back to their original form, then read the new json file through convert_json_to_questionnair.ipynb. This file creates the application. 

Application available at:
Emergency Surgery Project: MGH-All-Data/NSQIP/src_code/src_code_annita/OCT/oct_final_scripts/oct_emergency/results/questionnair_876_categorical_decoded.html


