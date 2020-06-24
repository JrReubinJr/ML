#logistic regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values

#split the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =  train_test_split(x,y)

from sklearn.preprocessing import StandardScaler
sx = StandardScaler(x_train)
sy = StandardScaler(y_train)
