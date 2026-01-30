# Import the core modules that we need as well as the desired dataset
from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt
import os

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


# Scaling all of our features between 0 and 1 is necessary 
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Briefly check the scaling on X_train
check_scaling = pd.DataFrame(X_train)
print(check_scaling.describe())

# Model uses a few input parameters that are not default. We are using a two layer network, a larger batch size, 
# and early_stopping will be used to prevent the model from overrunning our target.
# Originally we were using (10, 5) nodes, this was producing a very "blobby" scatterplot so I upped this to adjust.
mlp = MLPRegressor(random_state=42,
                   hidden_layer_sizes=(100,50),
                   max_iter=500, # Started with max_iter=200, increasing this to try to achieve convergence
                   batch_size=1000,
                   activation="relu",
                   validation_fraction=0.2,
                   early_stopping=True) # important!
mlp.fit(X_train, y_train)

# This will allow us to easily see whether or not we the model triggered the early_stopping feature. In this case it did and successfully found 
# convergence before reaching the iteration cap
if mlp.n_iter_ < mlp.max_iter:
    print(f"Stopped early, convergence found at {mlp.n_iter_} iterations.")
else:
    print("Reached maximum iterations.")

# Making predictions for our training data
train_predict = mlp.predict(X_train)


# Basic scatterplot to display the data, axis scales are between 0 and 5 since median house value is in units of $100,000
# Note that there is a vertical line right at 5. Researching the data a bit shows that the census capped house prices at $500,000 which creates this odd result.
plt.figure(figsize=(8,6))
plt.scatter(x=y_train, y=train_predict)
plt.plot([0,5],[0,5], '--k')
plt.axis('tight')
plt.xlabel('True price in $100,000s')
plt.ylabel('Predicted Price in $100,000s')
plt.suptitle('Training Results')

# Saving the figure before we display it.
if not os.path.exists('figures'):
    os.makedirs('figures')
plt.savefig('figures/training_scatterplot.png', dpi=300, bbox_inches='tight')

plt.show()

# Making predictions for our training data
test_predict = mlp.predict(X_test)


plt.figure(figsize=(8, 6))
plt.scatter(x=y_test, y=test_predict)
plt.plot([0, 5], [0, 5], '--k') # 45 degree line
plt.axis('tight')
plt.xlabel('True price ($100,000s)')
plt.ylabel('Predicted price ($100,000s)')
plt.suptitle('Test Results')

# Saving the figure before we display it.
plt.savefig('figures/test_scatterplot.png', dpi=300, bbox_inches='tight')
plt.show()