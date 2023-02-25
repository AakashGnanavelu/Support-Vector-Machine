# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 11:35:17 2021

@author: Aakash
"""

import pandas as pd
import numpy as np

data = pd.read_csv("fire.csv")
data.describe()

data.pop('month')
data.pop('day')

size_dict = {'size_category':   {'small': 0, 'large' : 1}}
data = data.replace(size_dict)

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

error_dict = {'model' : 'linear','poly','rbf','sigmoid',
              'error' : linear_error, poly_error, rbf_error, sigmoid_error}
