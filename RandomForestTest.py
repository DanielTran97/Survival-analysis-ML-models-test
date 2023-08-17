import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('framingham.csv')

#Clean data of un-needed column
df_cleaned = df.drop(columns =['education'])
df_cleaned = df_cleaned.dropna()
#feature selection
y = df_cleaned[["TenYearCHD"]]
X = df_cleaned.drop(columns =['TenYearCHD'])

#Convert y to a 1D array
#y = y.ravel()

#Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X) 

#Train test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state = 42)

#set RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

#Predict y
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy:.4f}")