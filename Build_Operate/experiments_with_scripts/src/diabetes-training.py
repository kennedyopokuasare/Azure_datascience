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
import mlflow

# No need to call mlflow.start_run becuause run is automatically started in a Job
mlflow.autolog()

mlflow.log_metric('mymetric', 1)
mlflow.log_metric('anothermetric',1)

