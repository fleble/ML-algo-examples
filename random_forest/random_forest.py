#!/usr/bin/python3


############################ RANDOM FOREST ############################


########################## READ ME ##########################
#
# DESCRIPTION
# *
#
# RESULTS
# *
#
#############################################################

## Load libairies
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn import metrics
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor


## Constant
RANDOM_STATE = np.random.randint(10000)
print("RANDOM_STATE = %d" %RANDOM_STATE)

FEATURES_CASE = 1
ENCODING_TYPE = 'mixed'  # 'OH' 'label' 'mixed'

## Read data
data = pd.read_csv('./input/home_train.csv')
data = data.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)  # shuffle data
if (FEATURES_CASE == 1):  # all features
    features = list(data.columns[1:80])
if (FEATURES_CASE == 2):  # Reduced features
    features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
target = ['SalePrice']

data = data[features+target]

print("\n===== RAW DATA =====")
print(data.head(10))


print("\n===== PREPROCESSING =====")

print("Dataset size before preprocessing: ", data.shape)
print("Features before preprocessing:\n",features)


print("\n=== MISSING VALUES ===")
# Get names of columns with missing values
cols_with_missing = [ cname for cname in features if data[cname].isnull().any() ]
print("Columns with missing values:\n\t", cols_with_missing)
#print("Excerp from the dataset:")
#print(data[cols_with_missing].head())

NaNfracThreshold = 0.15
cols_highNaNfrac = []
cols_lowNaNfrac = []
print("\nColumn\t\tNaN fraction")
for col in cols_with_missing:
    n_tab=2-int(len(col)/8)
    tab="\t"*n_tab
    NaNfrac = data[col].isnull().sum()/len(data[col])
    if (NaNfrac >= NaNfracThreshold): cols_highNaNfrac.append(col)
    else: cols_lowNaNfrac.append(col)
    print("%s%s%.2f" %(col, tab, NaNfrac) )

print("\nDrop columns with a NaN fraction greater than %s:" %NaNfracThreshold)
print(cols_highNaNfrac)
print("\nDrop entries with missing values")
data=data.drop(columns=cols_highNaNfrac)
data=data.dropna()

features = [ f for f in features if not(f in cols_highNaNfrac) ]


print("\n=== CATEGORICAL FEATURES ===")
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [ cname for cname in features if data[cname].dtype == "object" ]
cardinalityThreshold = 10
cols_highCard = []
cols_lowCard = []
print("Column\t\tCardinality")
for col in categorical_cols:
    n_tab=2-int(len(col)/8)
    tab="\t"*n_tab
    card = data[col].nunique()
    if ( card >= cardinalityThreshold ): cols_highCard.append(col)
    else: cols_lowCard.append(col)
    print("%s%s%d" %(col, tab, card) )

# Select numerical columns
numerical_cols = [cname for cname in features if data[cname].dtype in ['int64', 'float64']]
data_num = data[numerical_cols]


if (ENCODING_TYPE=='label'):
    print("\nTransform categorical features with label encoding")
    # Apply label encoder to each column with categorical data
    label_encoder = LabelEncoder()
    for col in categorical_cols:
        data[col] = label_encoder.fit_transform(data[col])


if (ENCODING_TYPE=='mixed'):
    print("\nTransform categorical features with mixed label and one-hot encoding:")
    print("One-Hot encoding for features with cardinality greater than ",cardinalityThreshold)
    print("\t",cols_highCard)
    print("Label encoding for other features")
    target_col = data['SalePrice']
    indices=data.index
    label_encoder = LabelEncoder()
    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    for col in cols_highCard:
        data[col] = label_encoder.fit_transform(data[col])
    data_label = data[cols_highCard]
    data_OH = pd.DataFrame(OH_encoder.fit_transform(data[cols_lowCard]))
    data_OH.index = indices  # One-hot encoding removed index; put it back

    # Add one-hot encoded columns to numerical features
    data = pd.concat([data_num, data_label, data_OH, target_col], axis=1)


print("\nDataset size after preprocessing: ", data.shape)

final_features = data.columns[:-1]
X = data[final_features]
y = data.SalePrice


## Check data
print("\n===== CHECK DATA =====")
print("\nFEATURES")
print(X.head())
print("\nTARGET")
print(y.head())


## Split train and test
print("\n===== SPLIT TRAIN TEST =====")
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.75, random_state=RANDOM_STATE)
print("Train set size: {}".format(X_train.shape[0]))
print("Test set size : {}".format(X_test.shape[0]))


## Training

score = "neg_mean_squared_error"
score2 = "RMSE"


## Dummy regressor
print("\n===== Dummmy Regressor (baseline) =====")
dum = DummyRegressor(strategy="mean")
print("Training...")
dum.fit(X_train,y_train)
print("\nPredictions:")
dum_mae = metrics.mean_absolute_error(dum.predict(X_test), y_test)
dum_mse = metrics.mean_squared_error(dum.predict(X_test), y_test)
print("\t MAE:  %.2e" %dum_mae)
print("\t MSE:  %.2e" %dum_mse)
print("\t RMSE: %.3f" %np.sqrt(dum_mse))


## Find the best RandomForestRegressor using GridSearchCV
print("\n===== Random Forest =====")
bootstrap_grid = [False, True]
max_features_grid = [0.05*i for i in range(1,9)]
n_estimators_grid = [10,50,100,200,500]
#criterion_grid = ["mse", "mae"]

param_grid = { #"bootstrap": bootstrap_grid,
               "max_features": max_features_grid,
               "n_estimators": n_estimators_grid
               }

rf = model_selection.GridSearchCV( RandomForestRegressor( #n_estimators=100,
                                                         random_state=RANDOM_STATE),
                                    param_grid = param_grid,
                                    cv = 5,
                                    scoring = score )

print("Training...")
rf.fit(X_train,y_train)

print("Best param: ", rf.best_params_)
print("Results of the cross validation:")
for mean, std, params in zip(
            rf.cv_results_['mean_test_score'], # mean score
            rf.cv_results_['std_test_score'],  # std score
            rf.cv_results_['params']           # hyperparameter value
            ):
    print("\t%s = %.2e (+/-%.2e) for %r" % (score, mean, 2*std, params))


print("\nPredictions:")
rf_mae = metrics.mean_absolute_error(rf.predict(X_test), y_test)
rf_mse = metrics.mean_squared_error(rf.predict(X_test), y_test)
print("\t MAE:  %.2e" %rf_mae)
print("\t MSE:  %.2e" %rf_mse)
print("\t RMSE: %.3f" %np.sqrt(rf_mse))

