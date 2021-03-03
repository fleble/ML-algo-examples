#!/usr/bin/python3


######################## MISSING VALUES TUTORIAL ########################

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer


# Function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid,n_est):
    model = RandomForestRegressor(n_estimators=n_est)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)
    

# Load the data
data = pd.read_csv('./input/melb_data.csv')

# Select target
y = data.Price

# To keep things simple, we'll use only numerical predictors
melb_predictors = data.drop(['Price'], axis=1)
X = melb_predictors.select_dtypes(exclude=['object'])

#print(melb_predictors.head())
#print(X.head())

# Divide data into training and validation subsets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=0)

# Get names of columns with missing values
cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]


### APPROACH 1: DROP COLUMNS WITH MISSING VALUES
# Drop columns in training and validation data
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)


### APPROACH 2: IMPUTATION - REPLACE MISSING VALUES BY
#               THE AVERAGE VALUE OF THE FEATURE
# Imputation
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# Imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns


### APPROACH 3: EXTENDED IMPUTATION
#               IMPUTATION PLUS ADDITION OF A BOLEAN COLUMN FOR EACH COLUMN
#               WITH NULLS TO KEEP TRACK OF WHICH VALUE WERE IMPUTED
# Make copy to avoid changing original data (when imputing)
X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()

# Make new columns indicating what will be imputed
for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

# Imputation
my_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))

# Imputation removed column names; put them back
imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_valid_plus.columns = X_valid_plus.columns

for n_est in [50,100,150,200,300]:
    print("n_estimators={}".format(n_est))
    
    print("  MAE from Approach 1 (Drop columns with missing values):")
    print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid, n_est))

    print("  MAE from Approach 2 (Imputation):")
    print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid, n_est))
    
    print("  MAE from Approach 3 (An Extension to Imputation):")
    print(score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid, n_est))
    print("\n")
