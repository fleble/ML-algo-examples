#!/usr/bin/python3


############################# XGBOOST TUTORIAL #############################


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# Read the data
data = pd.read_csv('./input/melb_data.csv')

# Select subset of predictors
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]

# Select target
y = data.Price

# Separate data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=1)

# Training with default parameters
model = XGBRegressor()
model.fit(X_train, y_train)
predictions = model.predict(X_valid)
print("\nDEFAULT PARAMETERS")
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)) +"\n")

### EARLY_STOPPING_ROUNDS
# Training untill n_estimators cycles are done or when the xgboost score does not
# improve for early_stopping_rounds rounds
model = XGBRegressor(n_estimators=1000)
model.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_valid, y_valid)],
          verbose=False)
predictions = model.predict(X_valid)
print("\nEARLY STOPPING ROUNDS")
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)) +"\n")


### LEARNING_RATE
model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
model.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_valid, y_valid)],
          verbose=False)
predictions = model.predict(X_valid)
print("\nLEARNING RATE")
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)) +"\n")

