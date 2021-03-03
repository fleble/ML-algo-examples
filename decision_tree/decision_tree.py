#!/usr/bin/python3


############################ DECISION TREE ############################

import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)
    

# Path of the file to read
iowa_file_path = './input/home_train.csv'

home_data = pd.read_csv(iowa_file_path)
# Create target object and call it y
y = home_data.SalePrice
# Create X
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Specify Model
iowa_model = DecisionTreeRegressor(random_state=1)
# Fit Model
iowa_model.fit(train_X, train_y)

# Make validation predictions and calculate mean absolute error
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
#print("Validation MAE: {:,.0f}".format(val_mae))
print("Validation MAE: {}".format(val_mae))


# Write loop to find the ideal tree size from candidate_max_leaf_nodes
candidate_max_leaf_nodes = [5, 25, 50, 60, 65, 70, 80, 100]
mae_list=[get_mae(max_leaves, train_X, val_X, train_y, val_y) for max_leaves in candidate_max_leaf_nodes]
best_tree_size = candidate_max_leaf_nodes[np.argmin(mae_list)]
print("Best max leaf nodes: {}".format(best_tree_size))
