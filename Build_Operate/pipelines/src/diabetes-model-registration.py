import sklearn
from azureml.core import Model
import argparse

# Set regularization hyperparameter
parser = argparse.ArgumentParser()
parser.add_argument('--reg-rate', type=float, dest='reg_rate', default=0.01)
parser.add_argument('--test-size', type=float, dest='test_size', default=0.30)
args = parser.parse_args()

filename = 'outputs/model.pkl'
Model.register(
    workspace = ws,
    model_name="diabetes-classification-model",
    model_path = filename,
    description = "A LogisticRegression classification model for Diabetes",
    tags = { 'data-format':"CSV", "regularization-rate":args.reg_rate, "test-size":args. test_size},
    model_framework = Model.Framework.SCIKITLEARN,
    model_framework_version = str(sklearn.__version__)
)
