# My_ML-Projects
Other Data science projects are uploaded.


###GradientBoostingRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, mean_absolute_percentage_error

# Load dataset
df = pd.read_csv('HousingData.csv')

# Fill missing values with a specific value (e.g., 0)
df = df.fillna(0)

# Independent variables (iv) and dependent variable (dv)
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

def metrics_GradientBoostingRegressor():
    print('GradientBoostingRegressor Metrics Score:')
    print(f"Mean Squared Error : {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R - Squared Score: {r2}")
    print(f"Explained Variance Score: {evs}")
    print(f"Mean Absolute Percentage Error: {mape}")
    print('----------------------------------------------------')

metrics_GradientBoostingRegressor() 

#MULTIPLE LINEAR REGRESSION
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, mean_absolute_percentage_error

# Load dataset
df = pd.read_csv('HousingData.csv')

# Fill missing values with a specific value (e.g., 0)
df = df.fillna(0)

# Independent variables (iv) and dependent variable (dv)
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the  model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

def metrics_linear():
    print('Linear Metrics Score:')
    print(f"Mean Squared Error : {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R - Squared Score: {r2}")
    print(f"Explained Variance Score: {evs}")
    print(f"Mean Absolute Percentage Error: {mape}")
    print('----------------------------------------------------')

metrics_linear()   

#Extreme Gradient Boosting. REGRESSION
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, mean_absolute_percentage_error

# Load dataset
df = pd.read_csv('HousingData.csv')

# Fill missing values with a specific value (e.g., 0)
df = df.fillna(0)

# Independent variables (iv) and dependent variable (dv)
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from xgboost import XGBRegressor

model = XGBRegressor(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

def metrics_xgb():
    print('xgb Metrics Score:')
    print(f"Mean Squared Error : {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R - Squared Score: {r2}")
    print(f"Explained Variance Score: {evs}")
    print(f"Mean Absolute Percentage Error: {mape}")
    print('----------------------------------------------------')

metrics_xgb()    

# POLYNOMIAL REGRESSION
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, mean_absolute_percentage_error

# Load dataset
df = pd.read_csv('HousingData.csv')

# Fill missing values with a specific value (e.g., 0)
df = df.fillna(0)

# Independent variables (iv) and dependent variable (dv)
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Polynomial Features
poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train)  
X_poly_test = poly.transform(X_test)        

# Train the Polynomial Regression model
model = LinearRegression()
model.fit(X_poly_train, y_train)

# Predict on the test set
y_pred = model.predict(X_poly_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

def metrics_poly():
    print('Polynomial Metrics Score:')
    print(f"Mean Squared Error : {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R - Squared Score: {r2}")
    print(f"Explained Variance Score: {evs}")
    print(f"Mean Absolute Percentage Error: {mape}")
    print('----------------------------------------------------')

metrics_poly()

#SVR - linear
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, mean_absolute_percentage_error
# Load dataset
df = pd.read_csv('HousingData.csv')

# Fill missing values with a specific value (e.g., 0)
df = df.fillna(0)

# Independent variables (iv) and dependent variable (dv)
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values
y = y.reshape(len(y), 1)  # Reshaping y to 2D array for scaling

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# Create a linear SVR model
model_svr = SVR(kernel='linear')

#Train the model
model_svr.fit(X_train_scaled, y_train_scaled.ravel()) # ravel - Flatten y to 1D array

# Make predictions
y_pred_scaled = model_svr.predict(X_test_scaled)
# Inverse transform predictions and actual values
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_test = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

def SVR_linear():
    print('SVR Linear Metrics Score:')
    print(f"Mean Squared Error : {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R - Squared Score: {r2}")
    print(f"Explained Variance Score: {evs}")
    print(f"Mean Absolute Precentage Error: {mape}")
    print('----------------------------------------------------')
SVR_linear() 

# SVR - Linear tuned
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, mean_absolute_percentage_error

# Load dataset
df = pd.read_csv('HousingData.csv')

# Fill missing values with a specific value (e.g., 0)
df = df.fillna(0)

# Independent variables (iv) and dependent variable (dv)
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values
y = y.reshape(len(y), 1)  # Reshaping y to 2D array for scaling

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train).flatten()
y_test_scaled = scaler_y.transform(y_test).flatten()

# Define the SVR model with a linear kernel
model_svr = SVR(kernel='linear')

# Define the parameter grid to search for best 'C' and 'epsilon' values
param_grid = {
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.1, 0.5, 1, 1.5, 2]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=model_svr, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)

# Train the model using GridSearchCV
grid_search.fit(X_train_scaled, y_train_scaled)

# Get the best parameters from GridSearchCV
best_params = grid_search.best_params_
print(f"Best Parameters from Grid Search: {best_params}")
best_score = -grid_search.best_score_
print(f"Best Mean Squared Error: {best_score}")

# Train the SVR model with the best parameters
best_svr = grid_search.best_estimator_

# Predict on test data
y_pred_model_svr = best_svr.predict(X_test_scaled)
y_pred_model_svr = scaler_y.inverse_transform(y_pred_model_svr.reshape(-1, 1))  # Reshaping y_pred for inverse transform


# Evaluate the tuned model
mse = mean_squared_error(y_test, y_pred_model_svr)
mae = mean_absolute_error(y_test, y_pred_model_svr)
r2 = r2_score(y_test, y_pred_model_svr)
evs = explained_variance_score(y_test, y_pred_model_svr)
mape = mean_absolute_percentage_error(y_test, y_pred_model_svr)

def SVR_tune_linear():
    print('SVR Linear Tuned Metrics Score:')
    print(f"Mean Squared Error : {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R - Squared Score: {r2}")
    print(f"Explained Variance Score: {evs}")
    print(f"Mean Absolute Percentage Error: {mape}")
    print('----------------------------------------------------')

SVR_tune_linear()
# SVR - Radial Basic Function
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, mean_absolute_percentage_error

# Load dataset
df = pd.read_csv('HousingData.csv')

# Fill missing values with a specific value (e.g., 0)
df = df.fillna(0)

# Independent variables (iv) and dependent variable (dv)
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values
y = y.reshape(len(y), 1)  # Reshaping y to 2D array for scaling

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardizing the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train).flatten()
y_test_scaled = scaler_y.transform(y_test).flatten()

# Define model with RBF kernel
model_svr = SVR(kernel='rbf')

# Train the model
model_svr.fit(X_train_scaled, y_train_scaled)

# Predict on test data
y_pred_model_svr = model_svr.predict(X_test_scaled)
y_pred_model_svr = scaler_y.inverse_transform(y_pred_model_svr.reshape(-1, 1))

# Evaluate the model
mse = mean_squared_error(y_test, y_pred_model_svr)
mae = mean_absolute_error(y_test, y_pred_model_svr)
r2 = r2_score(y_test, y_pred_model_svr)
evs = explained_variance_score(y_test, y_pred_model_svr)
mape = mean_absolute_percentage_error(y_test, y_pred_model_svr)

def SVR_rbf():
    print('SVR rbf Metrics Score:')
    print(f"Mean Squared Error : {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R - Squared Score: {r2}")
    print(f"Explained Variance Score: {evs}")
    print(f"Mean Absolute Percentage Error: {mape}")
    print('----------------------------------------------------')

SVR_rbf()

# SVR - Radial Basic Function tuned
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, mean_absolute_percentage_error

# Load dataset
df = pd.read_csv('HousingData.csv')

# Fill missing values with a specific value (e.g., 0)
df = df.fillna(0)

# Independent variables (iv) and dependent variable (dv)
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardizing the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# Define model with RBF kernel
model_rbf = SVR(kernel='rbf')

# Define parameter grid for RBF kernel
param_grid_rbf = {
    'C': [0.1, 1, 10, 100, 1000],
    'epsilon': [0.1, 0.5, 1, 1.5, 2],
    'gamma': ['scale', 'auto', 0.1, 0.01, 0.001]  
}

# Setup the grid search for RBF kernel
grid_search_rbf = GridSearchCV(model_rbf, param_grid_rbf, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)

# Fit grid search
best_model_rbf = grid_search_rbf.fit(X_train_scaled, y_train_scaled)

# Print the best parameters and best score
print("Best Parameters for RBF:", best_model_rbf.best_params_)
best_score = -best_model_rbf.best_score_
print(f"Best Score for rbf: {best_score}")

# Train the SVR model with the best parameters
best_svr = best_model_rbf.best_estimator_

# Predict on test data
y_pred_model_svr = best_svr.predict(X_test_scaled)
y_pred_model_svr = scaler_y.inverse_transform(y_pred_model_svr.reshape(-1, 1)).flatten()

# Evaluate the tuned model
mse = mean_squared_error(y_test, y_pred_model_svr)
mae = mean_absolute_error(y_test, y_pred_model_svr)
r2 = r2_score(y_test, y_pred_model_svr)
evs = explained_variance_score(y_test, y_pred_model_svr)
mape = mean_absolute_percentage_error(y_test, y_pred_model_svr)
#
    print('SVR rbf Tuned Metrics Score:')
    print(f"Mean Squared Error : {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R - Squared Score: {r2}")
    print(f"Explained Variance Score: {evs}")
    print(f"Mean Absolute Percentage Error: {mape}")
    print('----------------------------------------------------')

SVR_rbf_tuned()

#SVR -Polynomial
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, mean_absolute_percentage_error

# Load dataset
df = pd.read_csv('HousingData.csv')

# Fill missing values with a specific value (e.g., 0)
df = df.fillna(0)

# Independent variables (iv) and dependent variable (dv)
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values
y = y.reshape(len(y), 1)  # Reshaping y to 2D array for scaling

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardizing the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train).flatten()
y_test_scaled = scaler_y.transform(y_test).flatten()

# Define model with RBF kernel
model_svr = SVR(kernel='poly')

# Train the model
model_svr.fit(X_train_scaled, y_train_scaled)

# Predict on test data
y_pred_model_svr = model_svr.predict(X_test_scaled)
y_pred_model_svr = scaler_y.inverse_transform(y_pred_model_svr.reshape(-1, 1))

# Evaluate the model
mse = mean_squared_error(y_test, y_pred_model_svr)
mae = mean_absolute_error(y_test, y_pred_model_svr)
r2 = r2_score(y_test, y_pred_model_svr)
evs = explained_variance_score(y_test, y_pred_model_svr)
mape = mean_absolute_percentage_error(y_test, y_pred_model_svr)

def SVR_poly():
    print('SVR poly Metrics Score:')
    print(f"Mean Squared Error : {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R - Squared Score: {r2}")
    print(f"Explained Variance Score: {evs}")
    print(f"Mean Absolute Percentage Error: {mape}")
    print('----------------------------------------------------')

SVR_poly()

#SVR -Polynomial
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, mean_absolute_percentage_error

# Load dataset
df = pd.read_csv('HousingData.csv')

# Fill missing values with a specific value (e.g., 0)
df = df.fillna(0)

# Independent variables (iv) and dependent variable (dv)
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values
y = y.reshape(len(y), 1)  # Reshaping y to 2D array for scaling

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardizing the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train).flatten()
y_test_scaled = scaler_y.transform(y_test).flatten()

# Define model with RBF kernel
model_svr = SVR(kernel='poly')

# Train the model
model_svr.fit(X_train_scaled, y_train_scaled)

# Predict on test data
y_pred_model_svr = model_svr.predict(X_test_scaled)
y_pred_model_svr = scaler_y.inverse_transform(y_pred_model_svr.reshape(-1, 1))

# Evaluate the model
mse = mean_squared_error(y_test, y_pred_model_svr)
mae = mean_absolute_error(y_test, y_pred_model_svr)
r2 = r2_score(y_test, y_pred_model_svr)
evs = explained_variance_score(y_test, y_pred_model_svr)
mape = mean_absolute_percentage_error(y_test, y_pred_model_svr)

def SVR_poly():
    print('SVR poly Metrics Score:')
    print(f"Mean Squared Error : {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R - Squared Score: {r2}")
    print(f"Explained Variance Score: {evs}")
    print(f"Mean Absolute Percentage Error: {mape}")
    print('----------------------------------------------------')

SVR_poly()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, mean_absolute_percentage_error

# Load dataset
df = pd.read_csv('HousingData.csv')

# Fill missing values with a specific value (e.g., 0)
df = df.fillna(0)

# Independent variables (iv) and dependent variable (dv)
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values
y = y.reshape(len(y), 1)  # Reshaping y to 2D array for scaling

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardizing the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train).flatten()
y_test_scaled = scaler_y.transform(y_test).flatten()

# Define the model
model_svr = SVR()

# Define the hyperparameters grid for GridSearchCV
param_grid = {
    'kernel': ['poly'],
    'degree': [2, 3, 4],
    'C': [0.1, 1, 10],
    'epsilon': [0.01, 0.1, 0.5],
    'gamma': ['scale', 'auto']
}

# Apply GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(estimator=model_svr, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, verbose=2)

# Train the model with grid search
grid_search.fit(X_train_scaled, y_train_scaled)

# Best parameters from the grid search
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

# Predict on test data using the best model
best_model_svr = grid_search.best_estimator_
y_pred_model_svr = best_model_svr.predict(X_test_scaled)
y_pred_model_svr = scaler_y.inverse_transform(y_pred_model_svr.reshape(-1, 1))

# Evaluate the model
mse = mean_squared_error(y_test, y_pred_model_svr)
mae = mean_absolute_error(y_test, y_pred_model_svr)
r2 = r2_score(y_test, y_pred_model_svr)
evs = explained_variance_score(y_test, y_pred_model_svr)
mape = mean_absolute_percentage_error(y_test, y_pred_model_svr)

def SVR_poly():
    print('SVR poly Metrics Score:')
    print(f"Mean Squared Error : {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R - Squared Score: {r2}")
    print(f"Explained Variance Score: {evs}")
    print(f"Mean Absolute Percentage Error: {mape}")
    print('----------------------------------------------------')

SVR_poly()

import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, mean_absolute_percentage_error
#load datase
# Load dataset
df = pd.read_csv('HousingData.csv')

# Fill missing values with a specific value (e.g., 0)
df = df.fillna(0)

# Independent variables (iv) and dependent variable (dv)
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Initialize the Ridge regression model with alpha (lambda) value
ridge_reg = Ridge(alpha=1.0)

# Fit the model to the training data
ridge_reg.fit(X_train, y_train)
# Predict the target values for the test set
y_pred = ridge_reg.predict(X_test)
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

def metrics_Ridge():
    print('Ridge  Metrics Score:')
    print(f"Mean Squared Error : {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R - Squared Score: {r2}")
    print(f"Explained Variance Score: {evs}")
    print(f"Mean Absolute Percentage Error: {mape}")
    print('----------------------------------------------------')

metrics_Ridge()    

#ridge Tunned
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, mean_absolute_percentage_error

# Load dataset
df = pd.read_csv('HousingData.csv')

# Fill missing values with a specific value (e.g., 0)
df = df.fillna(0)

# Independent variables (iv) and dependent variable (dv)
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the model
ridge = Ridge()

# Define the hyperparameter grid to search
param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}

# Use GridSearchCV to search for the best hyperparameters
grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)

# Fit the model
grid_search.fit(X_train, y_train)

# Best hyperparameters
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

# Predict on the test set using the best model
best_ridge = grid_search.best_estimator_
y_pred = best_ridge.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

# Print metrics
def metrics_Ridge_tunned():
    print('Ridge Tunned Metrics Score:')
    print(f"Mean Squared Error : {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R - Squared Score: {r2}")
    print(f"Explained Variance Score: {evs}")
    print(f"Mean Absolute Percentage Error: {mape}")
    print('----------------------------------------------------')

metrics_Ridge_tunned()


import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, mean_absolute_percentage_error
#load datase
# Load dataset
df = pd.read_csv('HousingData.csv')

# Fill missing values with a specific value (e.g., 0)
df = df.fillna(0)

# Independent variables (iv) and dependent variable (dv)
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Initialize the Lasso regression model with alpha (lambda) value
lasso_reg = Lasso(alpha=1.0)

# Fit the model to the training data
lasso_reg.fit(X_train, y_train)
# Predict the target values for the test set
y_pred = lasso_reg.predict(X_test)
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

def metrics_Lasso():
    print('Lasso Metrics Score:')
    print(f"Mean Squared Error : {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R - Squared Score: {r2}")
    print(f"Explained Variance Score: {evs}")
    print(f"Mean Absolute Percentage Error: {mape}")
    print('----------------------------------------------------')

metrics_Lasso()    
#Lasso Tunned
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, mean_absolute_percentage_error

# Load dataset
df = pd.read_csv('HousingData.csv')

# Fill missing values with a specific value (e.g., 0)
df = df.fillna(0)

# Independent variables (iv) and dependent variable (dv)
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the model
lasso = Lasso()

# Define the hyperparameter grid to search
param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}

# Use GridSearchCV to search for the best hyperparameters
grid_search = GridSearchCV(estimator=lasso, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)

# Fit the model
grid_search.fit(X_train, y_train)

# Best hyperparameters
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

# Predict on the test set using the best model
best_lasso = grid_search.best_estimator_
y_pred = best_lasso.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

# Print metrics
def metrics_lasso_tunned():
    print('Lasso Tunned Metrics Score:')
    print(f"Mean Squared Error : {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R - Squared Score: {r2}")
    print(f"Explained Variance Score: {evs}")
    print(f"Mean Absolute Percentage Error: {mape}")
    print('----------------------------------------------------')

metrics_lasso_tunned()

import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, mean_absolute_percentage_error

# Load dataset
df = pd.read_csv('HousingData.csv')

# Fill missing values with a specific value (e.g., 0)
df = df.fillna(0)

# Independent variables (iv) and dependent variable (dv)
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the ElasticNet regression model with alpha (lambda) value and l1_ratio
elasticnet_reg = ElasticNet(alpha=1.0, l1_ratio=0.5)

# Fit the model to the training data
elasticnet_reg.fit(X_train, y_train)

# Predict the target values for the test set
y_pred = elasticnet_reg.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

# Print evaluation metrics
def metrics_ElasticNet():
    print('ElasticNet Metrics Score:')
    print(f"Mean Squared Error : {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R - Squared Score: {r2}")
    print(f"Explained Variance Score: {evs}")
    print(f"Mean Absolute Percentage Error: {mape}")
    print('----------------------------------------------------')

metrics_ElasticNet()

### elasticnet tunned 
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, mean_absolute_percentage_error

# Load dataset
df = pd.read_csv('HousingData.csv')

# Fill missing values with a specific value (e.g., 0)
df = df.fillna(0)

# Independent variables (iv) and dependent variable (dv)
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the ElasticNet model (without parameters for now)
elasticnet_reg = ElasticNet()

# Define the hyperparameter grid
param_grid = {
    'alpha': [0.01, 0.1, 1, 10, 100],  # Regularization strength
    'l1_ratio': [0.1, 0.5, 0.7, 1.0]  # Mix between L1 and L2
}

# Use GridSearchCV to search for the best combination of parameters
grid_search = GridSearchCV(estimator=elasticnet_reg, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)

# Fit the model to the training data
grid_search.fit(X_train, y_train)

# Best parameters found by GridSearchCV
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

# Predict on the test set using the best model
best_elasticnet = grid_search.best_estimator_
y_pred = best_elasticnet.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

# Print evaluation metrics
def metrics_ElasticNet():
    print('ElasticNet Tuuned Metrics Score:')
    print(f"Mean Squared Error : {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R - Squared Score: {r2}")
    print(f"Explained Variance Score: {evs}")
    print(f"Mean Absolute Percentage Error: {mape}")
    print('----------------------------------------------------')

metrics_ElasticNet()

Unsupervised learning 


import os
os.environ['OMP_NUM_THREADS'] = '1'

from sklearn.cluster import KMeans

# Set n_init explicitly to suppress the FutureWarning
kmeans = KMeans(n_clusters=3, n_init=10)
kmeans.fit(X)  # Replace X with your data
### to find the best k value by elbow method:
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd

# Import your dataset (replace 'your_dataset.csv' with the path to your file)
df = pd.read_csv('wine-clustering.csv')

# Assuming you want to use specific features for clustering
# Replace 'Feature1', 'Feature2' with actual column names from your dataset
X = df[['Alcohol', 'Malic_Acid', 'Color_Intensity']].values

# List to store inertia values for each k
inertia = []

# Try k values from 1 to 10
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

from yellowbrick.cluster import KElbowVisualizer

model = KMeans(random_state=1)
visualizer = KElbowVisualizer(model, k=(2,10))

visualizer.fit(X)
visualizer.show()
plt.show()

from sklearn.metrics import silhouette_score

# List to store silhouette scores for each k
silhouette_scores = []

# Try k values from 2 to 10 (Silhouette Score is undefined for k=1)
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    silhouette_scores.append(silhouette_score(X, labels))

model = KMeans(random_state=1)
visualizer = KElbowVisualizer(model, k=(2,10), metric='silhouette')

visualizer.fit(X)
visualizer.show()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Import your dataset (replace 'your_dataset.csv' with the path to your file)
df = pd.read_csv('wine-clustering.csv')

# Assuming you want to use specific features for clustering
X = df[['Alcohol', 'Malic_Acid', 'Color_Intensity']].values

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans Clustering
kmeans = KMeans(n_clusters=3, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)
centroids = kmeans.cluster_centers_

# DBSCAN Clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Agglomerative Clustering
agglo = AgglomerativeClustering(n_clusters=3)
agglo_labels = agglo.fit_predict(X_scaled)

# Gaussian Mixture Model
gmm = GaussianMixture(n_components=3, random_state=0)
gmm_labels = gmm.fit_predict(X_scaled)

# Calculate Clustering Metrics
metrics = {
    "Clustering Algorithm": ["KMeans", "DBSCAN", "Agglomerative", "GMM"],
    "Silhouette Score": [
        silhouette_score(X_scaled, kmeans_labels),
        silhouette_score(X_scaled, dbscan_labels),
        silhouette_score(X_scaled, agglo_labels),
        silhouette_score(X_scaled, gmm_labels)
    ],
    "Calinski-Harabasz Index": [
        calinski_harabasz_score(X_scaled, kmeans_labels),
        calinski_harabasz_score(X_scaled, dbscan_labels),
        calinski_harabasz_score(X_scaled, agglo_labels),
        calinski_harabasz_score(X_scaled, gmm_labels)
    ],
    "Davies-Bouldin Index": [
        davies_bouldin_score(X_scaled, kmeans_labels),
        davies_bouldin_score(X_scaled, dbscan_labels),
        davies_bouldin_score(X_scaled, agglo_labels),
        davies_bouldin_score(X_scaled, gmm_labels)
    ]
}

# Display Clustering Metrics
metrics_df = pd.DataFrame(metrics).set_index("Clustering Algorithm").round(3)
print(metrics_df)

# Visualization of KMeans Clustering with Centroids
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_labels, cmap='viridis', label='Clusters')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('KMeans Clustering with Centroids')
plt.legend()

# Visualization of DBSCAN Clustering
plt.subplot(2, 2, 2)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=dbscan_labels, cmap='viridis')
plt.title('DBSCAN Clustering')

# Visualization of Agglomerative Clustering
plt.subplot(2, 2, 3)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=agglo_labels, cmap='viridis')
plt.title('Agglomerative Clustering')

# Visualization of GMM Clustering
plt.subplot(2, 2, 4)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=gmm_labels, cmap='viridis')
plt.title('Gaussian Mixture Model Clustering')

plt.tight_layout()
plt.show()

# Create a Dendrogram for Agglomerative Clustering
plt.figure(figsize=(10, 7))
Z = linkage(X_scaled, method='ward')
dendrogram(Z, color_threshold=0.7 * np.max(Z[:, 2]), above_threshold_color='gray')
plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Sample Index or Cluster Size')
plt.ylabel('Distance (Ward\'s linkage)')
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import GridSearchCV
from scipy.cluster.hierarchy import dendrogram, linkage

# Import your dataset
df = pd.read_csv('wine-clustering.csv')

# Assuming you want to use specific features for clustering
X = df[['Alcohol', 'Malic_Acid', 'Color_Intensity']].values

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans Hyperparameter Tuning
kmeans_params = {'n_clusters': [2, 3, 4, 5], 'n_init': [10, 20, 30]}
kmeans = KMeans()
kmeans_grid = GridSearchCV(kmeans, kmeans_params, cv=3)
kmeans_grid.fit(X_scaled)
best_kmeans = kmeans_grid.best_estimator_
kmeans_labels = best_kmeans.predict(X_scaled)
centroids = best_kmeans.cluster_centers_

# Define a custom scoring function using silhouette score
def silhouette_scorer(estimator, X):
    # Get the labels from the DBSCAN model
    labels = estimator.fit_predict(X)
    # Only compute silhouette score if there are more than 1 cluster
    if len(set(labels)) > 1:
        return silhouette_score(X, labels)
    else:
        return -1  # Return a low score if only 1 cluster

# Define your parameter grid for DBSCAN
dbscan_params = {
    'eps': [0.3, 0.5, 0.7],  # example values
    'min_samples': [3, 5, 7]
}

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define DBSCAN model
dbscan = DBSCAN()

# Create GridSearchCV object with the custom scoring function
dbscan_grid = GridSearchCV(dbscan, dbscan_params, cv=3, scoring=make_scorer(silhouette_scorer))

# Fit the model
dbscan_grid.fit(X_scaled)

# Get the best estimator and labels
best_dbscan = dbscan_grid.best_estimator_
dbscan_labels = best_dbscan.fit_predict(X_scaled)

print("Best parameters found: ", dbscan_grid.best_params_)

# Agglomerative Clustering (no hyperparameter tuning through GridSearchCV for linkage methods)
agglo = AgglomerativeClustering(n_clusters=3)
agglo_labels = agglo.fit_predict(X_scaled)

# Gaussian Mixture Model Hyperparameter Tuning
gmm_params = {'n_components': [2, 3, 4], 'covariance_type': ['full', 'tied', 'diag']}
gmm = GaussianMixture(random_state=0)
gmm_grid = GridSearchCV(gmm, gmm_params, cv=3)
gmm_grid.fit(X_scaled)
best_gmm = gmm_grid.best_estimator_
gmm_labels = best_gmm.predict(X_scaled)

# Calculate Clustering Metrics
metrics = {
    "Clustering Algorithm": ["KMeans (Tuned)", "DBSCAN (Tuned)", "Agglomerative", "GMM (Tuned)"],
    "Silhouette Score": [
        silhouette_score(X_scaled, kmeans_labels),
        silhouette_score(X_scaled, dbscan_labels),
        silhouette_score(X_scaled, agglo_labels),
        silhouette_score(X_scaled, gmm_labels)
    ],
    "Calinski-Harabasz Index": [
        calinski_harabasz_score(X_scaled, kmeans_labels),
        calinski_harabasz_score(X_scaled, dbscan_labels),
        calinski_harabasz_score(X_scaled, agglo_labels),
        calinski_harabasz_score(X_scaled, gmm_labels)
    ],
    "Davies-Bouldin Index": [
        davies_bouldin_score(X_scaled, kmeans_labels),
        davies_bouldin_score(X_scaled, dbscan_labels),
        davies_bouldin_score(X_scaled, agglo_labels),
        davies_bouldin_score(X_scaled, gmm_labels)
    ]
}

# Display Clustering Metrics
metrics_df = pd.DataFrame(metrics).set_index("Clustering Algorithm").round(3)
print(metrics_df)

# Visualization of Tuned KMeans Clustering with Centroids
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_labels, cmap='viridis', label='Clusters')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title(f'KMeans Clustering (Tuned: n_clusters={best_kmeans.n_clusters})')
plt.legend()

# Visualization of Tuned DBSCAN Clustering
plt.subplot(2, 2, 2)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=dbscan_labels, cmap='viridis')
plt.title(f'DBSCAN Clustering (Tuned: eps={best_dbscan.eps}, min_samples={best_dbscan.min_samples})')

# Visualization of Agglomerative Clustering
plt.subplot(2, 2, 3)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=agglo_labels, cmap='viridis')
plt.title('Agglomerative Clustering')

# Visualization of Tuned GMM Clustering
plt.subplot(2, 2, 4)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=gmm_labels, cmap='viridis')
plt.title(f'Gaussian Mixture Model (Tuned: n_components={best_gmm.n_components})')

plt.tight_layout()
plt.show()

# Create a Dendrogram for Agglomerative Clustering
plt.figure(figsize=(10, 7))
Z = linkage(X_scaled, method='ward')
dendrogram(Z, color_threshold=0.7 * np.max(Z[:, 2]), above_threshold_color='gray')
plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Sample Index or Cluster Size')
plt.ylabel('Distance (Ward\'s linkage)')
plt.tight_layout()
plt.show()

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score

# For K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
labels_kmeans = kmeans.fit_predict(X_scaled)
silhouette_kmeans = silhouette_score(X_scaled, labels_kmeans)
print(f'Silhouette Score for K-Means: {silhouette_kmeans}')

# For DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_scaled)
silhouette_dbscan = silhouette_score(X_scaled, labels_dbscan) if len(set(labels_dbscan)) > 1 else -1
print(f'Silhouette Score for DBSCAN: {silhouette_dbscan}')

# For Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=3)
labels_hierarchical = hierarchical.fit_predict(X_scaled)
silhouette_hierarchical = silhouette_score(X_scaled, labels_hierarchical)
print(f'Silhouette Score for Hierarchical Clustering: {silhouette_hierarchical}')


