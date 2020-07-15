# Random Forest Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('D:\\08804\Documents\ML\MLpractice\datasets\\breast_Cancer.csv')
#X = dataset.iloc[:, :-1].values
X_data = dataset[['Bland Chromatin','Uniformity of Cell Size','Uniformity of Cell Shape','Bare Nuclei','Single Epithelial Cell Size']]
X = X_data.values

y = dataset.iloc[:, -1].values

columns = X_data.columns

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Random Forest Classification model on the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
FI = classifier.feature_importances_
fiDict = {}
for i in range(columns.size-1):
    fiDict[columns[i]] = FI[i]

#print(fiDict)
sortedFI = sorted(fiDict.items(), key=lambda x:x[1])
print(sortedFI)
for tup in sortedFI:
    print(f"{tup[0]} : {tup[1]}")


#print(FI)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))
