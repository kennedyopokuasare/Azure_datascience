# import libraries
import os
import argparse
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# Parse job parameters
parser = argparse.ArgumentParser()
parser.add_argument('--reg-rate', type=float, dest='reg_rate', default=0.01)
parser.add_argument('--test-size', type=float, dest='test_size', default=0.30)
parser.add_argument('--data-set', type=str,dest="data")
args = parser.parse_args()

reg_rate = args.reg_rate
test_size = args.test_size
print("Test data size:", test_size)
print("Regularization rate:", reg_rate)

# load the diabetes dataset
print("Loading Data...")
diabetes = pd.read_csv(args.data, header=0)

print("num_samples:", diabetes.shape[0])
features = diabetes[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']]
print("num_features:", features.shape[1])
print("features:", features.columns.values)

# separate features and labels
X = features.values
y = diabetes['Diabetic'].values

# split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

# train a logistic regression model
print('Training a logistic regression model with regularization rate of', reg_rate)
model = LogisticRegression(C=1/reg_rate, solver="liblinear").fit(X_train, y_train)

# calculate accuracy
y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
print('Accuracy:', float(acc))
