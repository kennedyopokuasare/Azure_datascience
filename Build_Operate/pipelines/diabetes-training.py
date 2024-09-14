# import libraries
from azureml.core import Run, Model
import joblib
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# Get the experiment run context 
run = Run.get_context()

# Set regularization hyperparameter
parser = argparse.ArgumentParser()
parser.add_argument('--reg-rate', type=float, dest='reg_rate', default=0.01)
parser.add_argument('--test-size', type=float, dest='test_size', default=0.30)
args = parser.parse_args()

reg_rate = args.reg_rate
test_size = args.test_size

# load the diabetes dataset
print("Loading Data...")
diabetes = pd.read_csv('../../data/diabetes.csv')

# separate features and labels
X = diabetes[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']].values
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
run.log('Accuracy', float(acc))

# calculate AUC
y_scores = model.predict_proba(X_test)
auc = roc_auc_score(y_test,y_scores[:,1])
print('AUC: ' + str(auc))
run.log('AUC', float(auc))

# Save the model
filename = 'outputs/model.pkl'
os.makedirs('outputs', exist_ok=True)
joblib.dump(value=model, filename= filename)
