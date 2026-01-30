# Import the core modules that we need as well as the desired dataset
from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt

# Import sklearn functions required for splitting, normalization, 
# modeling, and calculating error metrics

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor

#Step 1: Read in data and split train/test partitions

# Load California Housing dataset and convert to dataframe
housing = fetch_california_housing(as_frame=True)
housing_df = housing.frame

# Check the dataframe to get a bearing on our variables
print(housing_df.head())
print("\n", housing_df.shape)

# Split variables into features X and target variable Y
Y = housing_df['MedHouseVal']
X = housing_df.drop('MedHouseVal', axis=1)

# Check the shape of our features and target variable just to make sure

print("\n" + "-" * 30)
print(X.shape)
print(Y.shape)
print("-" * 30)

# Split data into training and test partitions
X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    test_size = 0.2,
                                                    shuffle = True,
                                                    random_state = 35)