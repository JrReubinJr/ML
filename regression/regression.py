################################################
##Template for all basic Regression ML models##
################################################

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

###
### Importing the dataset
###
dataset = pd.read_csv('YOUR DATASET HERE.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


###optional reshape of the Dependent Variable for to transform row into column
y = y.reshape(len(y),1)

###
### Encoding categorical data
###
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
x[:, 3] = labelencoder.fit_transform(x[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()


###
### Splitting the dataset into the Training set and Test set
###
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

###
###Feature Scaling
###
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X_train = sc_x.fit_transform(x_train)
y_train = sc_y.fit_transform(y_train)

###
### Training the model on the Training set
###

#Multiple Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x_train)
regressor = LinearRegression()
regressor.fit(x_poly, y_train)

#SVR
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x_train, y_train)

#Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x_train, y_train)

#Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(x_train, y_train)


###
### Predicting
###

#Multiple Linear Regression
y_pred = regressor.predict(x_test)

#Polynomial Regression
y_pred = regressor.predict(poly_reg.transform(x_test))

#SVR
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(x_test)))

#Decision Tree
y_pred = regressor.predict(x_test)

#Random Forest
y_pred = regressor.predict(x_test)


np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
