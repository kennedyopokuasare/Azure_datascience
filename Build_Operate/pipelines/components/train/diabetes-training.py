# import libraries
import os
import argparse
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

def main():

    # Parse job parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--reg_rate', type=float, default=0.01)
    parser.add_argument('--train_data', type=str, help="path to train data")
    parser.add_argument('--test_data', type=str, help="path to test data")
    parser.add_argument("--trained_model", type=str,help="path to trained data")
    args = parser.parse_args()


    # load the diabetes dataset
    print("Loading Data...")
    train_df = pd.read_csv(args.train_data, header=0, index_col="PatientID")
    y_train = train_df.pop("Diabetic")
    x_train = train_df.values
    test_df = pd.read_csv(args.test_data, header=0, index_col="PatientID")
    y_test = test_df.pop("Diabetic")
    x_test = test_df.values

    print("num_train_samples:", x_train.shape[0])
    print("num_test_samples:", x_test.shape[0])
    print("num_features:", x_train.shape[1])
    print("features:", x_train.columns.values)


    # train a logistic regression model
    print('Training a logistic regression model with regularization rate of', args.reg_rate)
    model = LogisticRegression(C=1/args.reg_rate, solver="liblinear").fit(x_train, y_train)

    # calculate accuracy
    y_hat = model.predict(x_test)
    acc = np.average(y_hat == y_test)
    print('Accuracy:', float(acc))

    # calculate AUC
    y_scores = model.predict_proba(x_test)
    auc = roc_auc_score(y_test,y_scores[:,1])
    print('AUC:' + str(auc))

    # Save the model to file
    print("Saving model to file")
    filename = os.path.join(args.trained_model,"model.pkl")
    os.makedirs('outputs', exist_ok=True)
    with open(filename, 'wb') as file:
        pickle.dump(model,file)


if __name__ == "__main__":
    main()
