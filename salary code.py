# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 14:44:41 2021

@author: Aakash
"""

import pandas as pd
import numpy as np

data_train = pd.read_csv(r"C:\Users\Aakash\Desktop\AAKASH\Coding Stuff\Full Data Science\Support vector Machine\Assginment\train.csv")
data_test = pd.read_csv(r"C:\Users\Aakash\Desktop\AAKASH\Coding Stuff\Full Data Science\Support vector Machine\Assginment\test.csv")

data = data_train
data = data.append(data_test)

train_len = len(data_train)
test_len  = len(data_test)
data_len = train_len + test_len
true_data_len = len(data)
print(data_len,true_data_len)

data.describe()

data.columns = ['age', 'work', 'edu', 'edu_no','marry','job','relation','race','sex','gain','loss','hrs','contry','salary']

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

labelencoder = LabelEncoder()
enc = OneHotEncoder(handle_unknown='ignore')

data['work'] = labelencoder.fit_transform(data['work'])
enc_work_df = pd.DataFrame(enc.fit_transform(data[['work']]).toarray())

enc_work_df.columns = ['Federal_Gov','Local_Gov','Private','Self_emp_inc','Self_emp_not_inc','State_gov','Without_Pay']

data['edu'] = labelencoder.fit_transform(data['edu'])
enc_edu_df = pd.DataFrame(enc.fit_transform(data[['edu']]).toarray())

enc_edu_df.columns = ['10th','11th','12th','1st - 4th', '5th-6th','7th-8th','9th','Assoc-acdm','Assoc-voc','Bachelors','Doctorate','HS_grad','Masters','Preschool','Prof_School','Some_College']

data['marry'] = labelencoder.fit_transform(data['marry'])
enc_marry_df = pd.DataFrame(enc.fit_transform(data[['marry']]).toarray())

enc_marry_df.columns = ['Divorced','married-AF-spouce','married-civ-spouce','Marries-spose-absent','Never-married','seperated','Widowed']

data['job'] = labelencoder.fit_transform(data['job'])
enc_job_df = pd.DataFrame(enc.fit_transform(data[['job']]).toarray())

enc_job_df.columns = ['Adm_clerical','Armed_Forces','Craft_repair','Exec_managerial', 'farming_fishing', 'Handlers-cleaners','Machine-op-inspect','Other_Service','Priv-house-serv','Prof-specialty','Protective_sevices','Sales','Tech_Sepport','Transport_moving']

data['relation'] = labelencoder.fit_transform(data['relation'])
enc_relation_df = pd.DataFrame(enc.fit_transform(data[['relation']]).toarray())

enc_relation_df.columns = ['Husband','Not_in_family','Other_relative','Own_child', 'Unmaried', 'Wife']

data['race'] = labelencoder.fit_transform(data['race'])
enc_race_df = pd.DataFrame(enc.fit_transform(data[['race']]).toarray())

enc_race_df.columns = ['Amer_Indian_Eskimo','Asian_Pac_Islander','Black','Other','White']

data['sex'] = labelencoder.fit_transform(data['sex'])
enc_sex_df = pd.DataFrame(enc.fit_transform(data[['sex']]).toarray())

enc_sex_df.columns = ['Female','Male']

data['contry'] = labelencoder.fit_transform(data['contry'])
enc_country_df = pd.DataFrame(enc.fit_transform(data[['contry']]).toarray())

salary_dict = {'salary':   {' <=50K':0, ' >50K' :1}}
data = data.replace(salary_dict)

del data['work']
del data['edu']
del data['marry']
del data['job']
del data['relation']
del data['race']
del data['sex']
del data['contry']

data = data.join(enc_work_df)
data = data.join(enc_edu_df)
data = data.join(enc_marry_df)
data = data.join(enc_job_df)
data = data.join(enc_relation_df)
data = data.join(enc_race_df)
data = data.join(enc_sex_df)
data = data.join(enc_country_df)


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

train,test = train_test_split(data,test_size = 0.20)

train_X = train.iloc[:,:-1]
train_y = train.iloc[:,-1]
test_X  = test.iloc[:,:-1]
test_y  = test.iloc[:,-1]

model_linear = SVC(kernel = "linear")
model_linear.fit(train_X,train_y)
pred_test_linear = model_linear.predict(test_X)

linear_error = np.mean(pred_test_linear==test_y)

model_poly = SVC(kernel = "poly")
model_poly.fit(train_X,train_y)
pred_test_poly = model_poly.predict(test_X)

poly_error = np.mean(pred_test_poly==test_y)

model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X,train_y)
pred_test_rbf = model_rbf.predict(test_X)

rbf_error = np.mean(pred_test_rbf==test_y)

model_sigmoid = SVC(kernel = "sigmoid")
model_sigmoid.fit(train_X,train_y)
pred_test_sigmoid = model_sigmoid.predict(test_X)

sigmoid_error = np.mean(pred_test_sigmoid==test_y)

error_dict = {'model type':  ['linear','poly','rbf','sigmoid'],
              'error': [linear_error, poly_error, rbf_error, sigmoid_error]
        }

error_frame = pd.DataFrame (error_dict, columns = ['model type', 'error'])
