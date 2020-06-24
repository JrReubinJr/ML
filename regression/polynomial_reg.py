#Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x, y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)
poly_reg_model = LinearRegression()
poly_reg_model.fit(x_poly, y)

print(poly_reg_model.predict(poly_reg.fit_transform([[6.5]])))


#plotting the data
plt.scatter(x, y, color='red')
plt.plot(x, poly_reg_model.predict(x_poly), color='blue')
plt.title("Truth or Bluff (Poly Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()


# Splitting the dataset into the Training set and Test set
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.0, random_state = 0)

