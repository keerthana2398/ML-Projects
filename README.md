# ML-Projects
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



