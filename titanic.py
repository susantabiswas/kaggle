# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 19:22:12 2018
** Titanic Kaggle **
@author: SUSANTA
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#################### ################################## Training data ###################
# importing  dataset
dataset = pd.read_csv('train.csv')
# check the missing value count
dataset.apply(lambda x: x.count(), axis = 0)
# Age, cabin,embarked have missing values
X = dataset.iloc[:,[2,4,5,6,7,9]].values
y = dataset.iloc[:,1].values

############## Missing values ###################
# dealing with missing data
from sklearn.preprocessing import Imputer
# make an object of Imputer class
impute_obj = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
# now we will use the obj and feed it the data that needs imputation
imputer_obj = impute_obj.fit(X = X[:,2].reshape(-1,1), y=None)
# now replace the missing values
X[:,2] = np.squeeze(impute_obj.transform(X[:,2].reshape(-1,1)))



############# dealing with categorical data ################
# now to deal with columns which has non-numerical data,
# we call these categorical data ,as the data corresponds has some descrete set of values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# make an encoder object
encoder_1 = LabelEncoder()
# fit and transform the labels into encodings
X[:,1] = encoder_1.fit_transform(X[:,1])
# we will dummy variables for making sure that algo doesn't consider encoded no.
# as having relative importance to each other for this we will use one hot encoder
one_hot_enc_obj = OneHotEncoder(categorical_features = [1])
X = one_hot_enc_obj.fit_transform(X).toarray()
# avoid dummy trap
X = X[:,1:]
X_train = np.copy(X)
y_train = np.copy(y)

del X,y

#################################################### Test data ##################
dataset1 = pd.read_csv('test.csv')
# check the missing value count
dataset1.apply(lambda x: x.count(), axis = 0)
# Age, Fare, cabin,embarked have missing values
X = dataset1.iloc[:,[1,3,4,5,6,8]].values

############## Missing values ###################
# dealing with missing data
from sklearn.preprocessing import Imputer
# make an object of Imputer class
impute_obj = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
# now we will use the obj and feed it the data that needs imputation
imputer_obj = impute_obj.fit(X = X[:,[2,5]], y=None)
# now replace the missing values
X[:,[2,5]] = np.squeeze(impute_obj.transform(X[:,[2,5]]))



############# dealing with categorical data ################
# now to deal with columns which has non-numerical data,
# we call these categorical data ,as the data corresponds has some descrete set of values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# make an encoder object
encoder_1 = LabelEncoder()
# fit and transform the labels into encodings
X[:,1] = encoder_1.fit_transform(X[:,1])
# we will dummy variables for making sure that algo doesn't consider encoded no.
# as having relative importance to each other for this we will use one hot encoder
one_hot_enc_obj = OneHotEncoder(categorical_features = [1])
X = one_hot_enc_obj.fit_transform(X).toarray()
# avoid dummy trap
X = X[:,1:]
X_test = np.copy(X)

del X


######################### APPLYING MODEL #########################

### XGBoost
# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier(learning_rate = 0.1, n_estimators=100, reg_alpha = 0.06, reg_lambda = 0.009, gamma = 0.007)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print('Training Accuracy:'+str(accuracies.mean()))
print('Training SD:'+str(accuracies.std()))


############## Hypertunung
# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'learning_rate': [0.09, 0.1, 0.2], 'reg_alpha': [ 0.05, 0.06, 0.07],
               'reg_lambda': [0.007, 0.008, 0.009, 0.01], 'gamma': [0.007, 0.008, 0.006, 0.009, 0.01, 0.02]}
             ]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

################### Random Forest
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 300, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

################### making submission file ########################################
df = pd.read_csv('test.csv')
submission = df.iloc[:,0].values
submission = submission.reshape(-1,1).astype(int)
y_pred = y_pred.reshape(-1,1).astype(int)
submission = np.append(submission, y_pred, axis = 1).astype(dtype = int)
# save as csv
np.savetxt(r'C:\Users\SUSANTA\Desktop\kaggle\titanic_sub.csv', submission.astype(int), fmt='%i', delimiter=",")