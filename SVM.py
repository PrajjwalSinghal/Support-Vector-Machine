#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 17:05:06 2019

@author: prajjwalsinghal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib.pyplot
def getAccuracy(Y_test, Y_Pred):
    correct = 0;
    for x in range(len(Y_test)):
        if Y_test.iloc[x] == Y_Pred[x]:
            correct += 1
    return (correct/float(len(Y_test))) * 100.0


dataset = pd.read_csv('Iris.csv')
dataset.shape

# Dividing into dependent and independent variables
X = dataset.drop('Species', axis = 1)
Y = dataset['Species']

# separating into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)

#Training the algorithm
from sklearn.svm import SVC
model = SVC(kernel = 'linear')
model.fit(X_train, Y_train)

# Predicting
Y_Pred = model.predict(X_test)

from sklearn.metrics import classification_report

print(getAccuracy(Y_test, Y_Pred))

# Accuracy = 100.0