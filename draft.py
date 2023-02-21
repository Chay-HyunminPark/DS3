#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 10:57:40 2023

@author: yewonshome
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

# Read csv file
dataframe = pd.read_csv("Featurized Data.csv")

# There is no NA values, we proceed without dropping any NA values

# Glimpse of dataset
summary1 = dataframe.describe()
# This gives a general overview of the dataframe. 

# Remove [ and ] and make all numerical
dataframe["initial discharge capacity"] = dataframe["initial discharge capacity"].str.strip("[")
dataframe["initial discharge capacity"] = dataframe["initial discharge capacity"].astype(float)
dataframe["Delta_Variance"] = dataframe["Delta_Variance"].str.strip("]")
dataframe["Delta_Variance"] = dataframe["Delta_Variance"].astype(float)

# Split dataset into predictors and outcome variable
predictors = dataframe.drop(columns=["Remaining Useful Life"])


# Split dataset into training/test to fit the model
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, dataframe["Remaining Useful Life"], 
                                                    test_size=0.2)

# Standardizing the X predictors 
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(x_train)

x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Random Forest Model
from sklearn.ensemble import RandomForestRegressor
forest=RandomForestRegressor()
forest.fit(x_train_scaled, y_train)

# Check the performance of the model
forest_score = forest.score(x_test_scaled, y_test)
forest_score
# The R^2 is 0.8179, which means the predictors can explain approximately 82% of
# the variation in the outcome variable.

# Check for MSE
y_pred = forest.predict(x_test)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
# MSE is 0.025; it is close to 0 and it means the model's prediction is close to the actual value(test).