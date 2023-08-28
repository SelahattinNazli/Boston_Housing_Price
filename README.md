# Boston_Housing_Price_Prediction

## Overview:
This project focuses on predicting the median value of owner-occupied homes in the Boston area using various features related to housing conditions. The dataset was derived from the U.S. Census Service and contains information collected by the U.S. Census Service concerning housing in the area of Boston, Massachusetts.

## Dataset:
The dataset consists of 506 entries with the following features:

- **CRIM:** Per capita crime rate by town 

- **ZN:** Proportion of residential land zoned for lots over 25,000 sq. ft.

- **INDUS:** Proportion of non-retail business acres per town

- **CHAS:** Charles River dummy variable (1 if tract bounds river; 0 otherwise)

- **NOX:** Nitric oxide concentration (parts per 10 million)

- **RM:** Average number of rooms per dwelling

- **AGE:** Proportion of owner-occupied units built prior to 1940  

- **DIS:** Weighted distances to five Boston employment centers

- **RAD:** Index of accessibility to radial highways

- **TAX:** Full-value property tax rate per $10,000

- **PTRATIO:** Pupil-teacher ratio by town

- **LSTAT:** Percentage of lower status of the population

- **MEDV:** Median value of owner-occupied homes in $1000s (Target Variable)


  ## Methods and Models:
Various regression models were explored to predict the median value (MEDV) of homes, including:

- Linear Regression
- Decision Tree Regression
- Random Forest Regression
- Neural Network (MLP Regressor)
- Gradient Boosting Regressor
  
## Results:
Among the models, Random Forest Regressor and Gradient Boosting Regressor achieved the lowest Root Mean Squared Error (RMSE), indicating the best performance on the test data.

## Future Work:
Further improvements can be achieved through hyperparameter tuning, ensemble methods, and more advanced machine learning models.


# Importing the Libraries
```python
#Let's import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```
# Importing the Dataset
```python
# Column headers for the Boston Housing dataset
column_headers = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"
]

# Load the dataset with the column headers
housing_data = pd.read_csv("housing.csv", header=None, names=column_headers,delim_whitespace=True)
#Drop the "B" column from dataset
housing_data = housing_data.drop(columns=['B'])

# Display the first few rows of the dataset with proper column headers
housing_data.head()
```
# Exploratory Data Analysis (EDA)
```python
# Summary statistics for the entire dataset
summary_statistics = housing_data.describe()

summary_statistics
```

```python
#Check for missing values
missing_values = housing_data.isnull().sum()

missing_values
```

```python
# Display information of dataset
housing_data.info()
```

```python
# Initialize the figure for histograms and boxplots
plt.figure(figsize=(20, 18))

# Plotting histograms and boxplots for all the variables
for i, var in enumerate(housing_data, 1):
    plt.subplot(4, 6, i)
    sns.histplot(housing_data[var], bins=30, kde=True)
    plt.title(var)
    plt.xlabel('')
    
    # Plotting boxplots for the variables on the right
    plt.subplot(4, 6, i + 6)
    sns.boxplot(y=housing_data[var])
    plt.title(var + ' (Boxplot)')
    plt.ylabel('')

plt.tight_layout()
plt.show()
```
CRIM: The distribution is highly skewed to the right, with most towns having a low crime rate. There are a few towns with exceptionally high crime rates, which are likely outliers.  

RM: The distribution is close to normal, but there are some towns with unusually low or high average room numbers.  

MEDV: The distribution is approximately normal, but there's a noticeable spike at the value of 50.   
This could indicate possible capping or clipping of home values at that point. There are several towns that are identified as outliers, particularly on the higher end.
```python
# Correlation matrix for the entire dataset (without the 'B' column)
correlation_matrix_all = housing_data.corr()

# Plotting the correlation matrix as a heatmap
plt.figure(figsize=(12, 9))
sns.heatmap(correlation_matrix_all, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
plt.title('Correlation Heatmap for All Variables')
plt.show()
```
CRIM (Crime Rate) and MEDV (Median Value of Homes): -0.39
As the crime rate increases, the median value of homes tends to decrease. This is a negative correlation, indicating that areas with higher crime rates are associated with lower house values. 

RM (Average Number of Rooms) and MEDV (Median Value of Homes): 0.7
There's a strong positive correlation between the average number of rooms and the median value of homes. This means houses with more rooms, on average, tend to be more valuable. 

LSTAT (Lower Status of the Population) and MEDV (Median Value of Homes): 0.74
There's a strong negative correlation between the percentage of the lower status of the population and the median value of homes. This suggests that areas with a higher proportion of lower-status residents tend to have lower house values.
 
The heatmap gives a clear visual representation of the relationships between the variables. High positive values (closer to 1) indicate a strong positive correlation, while high negative values (closer to -1−1) indicate a strong negative correlation. Values close to 0 suggest weak or no correlation.
# Pre-processing Steps
We'll scale the features so they have a mean of 0 and a standard deviation of 1. This is important for regression to ensure all features have equal weightage. Then, let's split the data into training and testing sets.
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Separate features and target variable
X = housing_data.drop("MEDV", axis=1)
y = housing_data["MEDV"]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

X_train.shape, X_test.shape
```
The data has been successfully preprocessed and split:

Training set: 404 samples
Testing set: 102 samples

# Regression Model Selection
```python
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np

# Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
    "Random Forest Regressor": RandomForestRegressor(random_state=42)
}

# Train models and evaluate on the test set
rmse_results = {}
r2_results = {}
for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_results[name] = rmse
    r2 = r2_score(y_test,y_pred)
    r2_results[name] = r2
rmse_results,\
r2_results
```
Here are the Root Mean Squared Error (RMSE) results and R^2 scores for the regression models:

Linear Regression: RMSE = 4.77
Decision Tree Regressor: RMSE = 2.99
Random Forest Regressor: RMSE = 2.84

From the results, the Random Forest Regressor has the lowest RMSE, making it the best-performing model among the ones tested on this dataset. It's followed closely by the Decision Tree Regressor. Both tree-based models outperformed the linear models in this case.

```python
from sklearn.neural_network import MLPRegressor

# Initialize the neural network
nn_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42, activation='relu', solver='adam')

# Train the model
nn_model.fit(X_train, y_train)

# Predict on the test set
y_pred_nn = nn_model.predict(X_test)

# Calculate RMSE and R^2 for the neural network
rmse_nn = np.sqrt(mean_squared_error(y_test, y_pred_nn))
r2_nn = r2_score(y_test, y_pred_nn)

rmse_nn, r2_nn
```
The neural network model achieved an RMSE of 3.53, which is an improvement over the linear models but is slightly higher than the Random Forest Regressor. The R^2 value of 0.83 indicates that the model explains approximately 83% of the variance in the test set.
# Regularizations
```python
from sklearn.neural_network import MLPRegressor

# Initialize the neural network
nn_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42, activation='relu', solver='adam')

# Train the model
nn_model.fit(X_train, y_train)

# Predict on the test set
y_pred_nn = nn_model.predict(X_test)

# Calculate RMSE and R^2 for the neural network
rmse_nn = np.sqrt(mean_squared_error(y_test, y_pred_nn))
r2_nn = r2_score(y_test, y_pred_nn)

rmse_nn, r2_nn
```
The Gradient Boosting model with polynomial features and hyperparameter tuning achieved an RMSE of 3.11, which is an improvement over the Neural Network and linear models, and is close to the Random Forest Regressor's performance. The R^2 value of 0.87 indicates that the model explains approximately 87% of the variance in the test set.
# Visualisation of RMSE Results
```python
# Re-initializing the models

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

# Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))

# Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

# Neural Network
nn_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42, activation='relu', solver='adam')
nn_model.fit(X_train, y_train)
y_pred_nn = nn_model.predict(X_test)
rmse_nn = np.sqrt(mean_squared_error(y_test, y_pred_nn))

# Gradient Boosting with simplified hyperparameters
simple_param_grid = {
    'n_estimators': [50, 100],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 4],
}
simple_grid_search = GridSearchCV(GradientBoostingRegressor(random_state=42), simple_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
simple_grid_search.fit(X_train_poly, y_train)
best_gbr_simple = simple_grid_search.best_estimator_
y_pred_gbr_simple = best_gbr_simple.predict(X_test_poly)
rmse_gbr_simple = np.sqrt(mean_squared_error(y_test, y_pred_gbr_simple))

# Collating the RMSE results for visualization
rmse_values = {
    "Linear Regression": rmse_lr,
    "Decision Tree": rmse_dt,
    "Random Forest": rmse_rf,
    "Neural Network": rmse_nn,
    "Gradient Boosting": rmse_gbr_simple
}

# Sorting the RMSE values for better visualization
sorted_rmse = dict(sorted(rmse_values.items(), key=lambda item: item[1]))

# Plotting the RMSE values
plt.figure(figsize=(12, 8))
plt.barh(list(sorted_rmse.keys()), list(sorted_rmse.values()), color='skyblue')
plt.xlabel('RMSE')
plt.title('Comparison of RMSE for Different Models')
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.show()
```
