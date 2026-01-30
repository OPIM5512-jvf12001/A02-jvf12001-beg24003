# Import the core modules that we need as well as the desired dataset
from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

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
                   # I tried adding a 3rd layer of 25 nodes and the error metrics actually got worse.
                   # shrinking layer 2 to 25 nodes seems to be the sweet spot.
                   hidden_layer_sizes=(100, 25),
                   max_iter=1000, # Started with max_iter=200, increasing this to try to achieve convergence
                   batch_size=500,
                   # I tried using tanh instead and it took much longer to converge and produced worse results.
                   activation="relu",
                   validation_fraction=0.2,
                   # Tightening the tolerance improved metrics a little, further changes do not help.
                   tol=1e-5,
                   # Increasing the alpha showed some improved in the error metrics, futher increases make things worse.
                   alpha = 0.001,
                   # I also tried a different solver but the default adam performed better.
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


# Basic scatterplot to display the data, axis scales are between 0 and 6 since median house value is in units of $100,000.
# Note that there is a vertical line right at 5. Researching the data a bit shows that the census capped house prices at $500,000 which creates this odd result.
# Axes are expanded to (0,6) since the model could still predict a value greater than 5 even if the actual values will never be that large.
sns.set_theme(style='ticks')
plt.figure(figsize=(8,7))
#Using Seaborn instead of matplotlib to get a little more creative with our plots.
train_plot = sns.scatterplot(x=y_train, y=train_predict, color = "#004d4d", s = 20, alpha = 0.4, edgecolor = 'w')
plt.plot([0,6],[0,6], color = 'darkorange', linewidth = 2)
plt.axis('tight')
plt.xlabel('True price in $100,000s', fontsize = 14, fontweight = 'bold')
plt.ylabel('Predicted Price in $100,000s', fontsize = 14, fontweight = 'bold')
plt.title('Training Results', pad = 30, fontsize = 18, fontweight = 'bold')

# Saving the figure before we display it.
if not os.path.exists('figures'):
    os.makedirs('figures')
plt.savefig('figures/training_scatterplot.png', dpi=300, bbox_inches='tight')

plt.show()

# Making predictions for our test data
test_predict = mlp.predict(X_test)


plt.figure(figsize=(8,7))
#Using Seaborn instead of matplotlib to get a little more creative with our plots.
test_plot = sns.scatterplot(x=y_test, y=test_predict, color = "#004d4d", s = 20, alpha = 0.4, edgecolor = 'w')
plt.plot([0,6],[0,6], color = 'darkorange', linewidth = 2)
plt.axis('tight')
plt.xlabel('True price in $100,000s', fontsize = 14, fontweight = 'bold')
plt.ylabel('Predicted Price in $100,000s', fontsize = 14, fontweight = 'bold')
plt.title('Test Results', pad = 30, fontsize = 18, fontweight = 'bold')

# Saving the figure before we display it.
plt.savefig('figures/test_scatterplot.png', dpi=300, bbox_inches='tight')
plt.show()



# Displaying our three important error metrics. The initial run of the model produces decent results and the metrics do not shift much between
# training and test data. This tells us that the model is not overfitting and it generalizing well.
print("\n" + "-" * 30)
print("Training R2 Score: ", r2_score(y_train, train_predict))
print("Test Data R2 Score: ", r2_score(y_test, test_predict))
print("\n" + "-" * 30)
print("Training MAE: ", mean_absolute_error(y_train, train_predict))
print("Test Data MAE: ", mean_absolute_error(y_test, test_predict))
print("\n" + "-" * 30)
print("Training MSE: ", mean_squared_error(y_train, train_predict))
print("Test Data MSE: ", mean_squared_error(y_test, test_predict))
print("\n" + "-" * 30)


