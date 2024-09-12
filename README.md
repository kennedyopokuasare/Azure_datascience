
# Machine Learning with Microsoft Azure

**Microsoft Azure Machine Learning** provides the cloud infrastructure, DataStores, scalable and on-demand compute, machine learning workflows for production-grade Machine Learning. Azure offers tools for the full lifecycle of Machine Learning, including the productionization and monitoring of models.

In previous Machine Learning work [[1](https://github.com/kennedyopokuasare/IBM_datascience), [2](https://github.com/kennedyopokuasare), [3](https://github.com/kennedyopokuasare/carat-analysis)], I have executed entire machine learning pipelines, from data cleaning to prediction, using my personal computer, Jupyter Notebook, and [Snakemake for workflow management](https://snakemake.readthedocs.io/en/stable/). There are bottlenecks when training machine learning models with large datasets on a personal computer with limited resources, especially when multiple experiments need to be conducted — a case in point is when I trained [five machine learning algorithms to predict the depression status of participants](https://www.sciencedirect.com/science/article/pii/S1574119222000566) as part of my PhD research.

This repository presents a project where I use Microsoft Azure Machine Learning Cloud services for training, inferencing and Productionization of Machine Learning Models.

## Exploratory Data Analysis

In `Azure ML Studio`, notebooks can be run just like you would on your computer. First, you need to provision a `compute instance` that will run the computations in the notebook.

For instance:

* I explored the student grades and hours of study dataset and examined the distribution of the students' grades. [Source](./With_code/02-visualize-data.ipynb)
* I used the K-Means clustering algorithm to cluster the seeds dataset based on two principal components of the dataset. [Source](./With_code/01-clustering-introduction.ipynb)

## Automated Machine Learning in Azure ML studio

`Automated Machine Learning` allows for automated training of multiple machine learning models, in parallel to predict a target, without the need to write code. AutoML selects the best models and data transformations based on a the estimator's metrics.

### AutoML completed, with MaxAbsScaler and XGBoostRegressor

In the screenshot, the AutoML selected the best model `XGBoostRegressor` with `MaxAbsScaler`based on metrics such as the `Normalized root mean square error`. The registered and deployed with a Real time endpoint.

<img src="./AutoML/1.%20AutoML%20completed.png" alt="drawing" width="1200"/>

### Explaining the AutoML regression model

For model explanability, the AutoML generates the feature importance for all features. It is possible to show the best K features usign the UI. In the screen shot, `workingday`, `temperation`, `year` and `humidity` were the best 4 features that explained the prediction of bicyle rentals.  

<img src="./AutoML/2. Explanations and feature importance.png" alt="drawing" width="1200"/>

Several metric to access the performance of the model is generated by AutoML. The AutoML also generates plots such as the Redisual plots for Regression models that can be use to qualitatively access the model's performance.

<img src="./AutoML/5. Evaluation Metrics.png" alt="drawing" width="1200"/>

## Designing ML pipelines with low code drag and drop UI

### Setup the datasource

<img src="./ML_Designer/Classification/0. Uploaded Dataset.png" alt="drawing" width="1200"/>

### Design classification pipeline

#### Designing a Logistic Regression classifier

<img src="./ML_Designer/Classification/1. Clasification pipeline.png" alt="drawing" width="1200"/>


#### Comparing Logistic Regression and Decision Forest classifiers

<img src="./ML_Designer/Classification/2. Compare two class logistic regression and Decision Forest.png" alt="drawing" width="1200"/>

#### Evaluation metrics of the Logistic Regression and Decision Forest classifiers

<img src="./ML_Designer/Classification/2.1 Metrics of two models.png" alt="drawing" width="1200"/>

### Inferencing with the trained model

<img src="./ML_Designer/Classification/3. Inference with Decision Forest classifier.png" alt="drawing" width="1200"/>

### Deploy model and test

```python
import requests
import json

endpoint = "<END-POINT-URL-HERE>"
key = "<API-KEY-HERE>"

payload = {
    "Inputs": {
        "input1":
        [
            {
                'Age':43,
                'PatientID': 1882185,
                'Pregnancies': 9,
                'Glucose': 104,
                'BloodPressure': 51,
                'SkinThickness': 7,
                'Insulin': 24,
                'BMI': 27.36983156,
                'DiabetesPedigreeFunction': 1.3504720469999998
            },
        ],
    },
}

#Set the content type and authentication for the request 
headers = {"Content-Type":"application/json", "Authorization":"Bearer " + key} 
input_json = json.dumps(payload)

#Send the request 
response = requests.post(endpoint, input_json, headers=headers) 
response.text

if response.status_code == 200: 
    y = response.json()  
    output = y["Results"]["WebServiceOutput0"][0]
    print("Predictions:") 
    print('Patient: {}\nPrediction: {}\nProbability: {:.2f}'.format(
        output["PatientID"],
        output["DiabetesPrediction"],
        output["Probability"]
    ))
else: 
    print(response) 
```

```txt
Predictions:
Patient: 1882185.0
Prediction: 1.0
Probability: 0.75
```

## Building and Operating Experiments

In most cases, production systems intergrate Azure Machine Learning into existing code bases. Therefore, the ML Designer and AutoML may not be the ideal choice in such instances. The Azure Machine Learning Python SDK provides creating scripts that can be use build and operate Machine Learning systems. For aditional details, see the [What is the Azure Machine Learning SDK for Python](https://learn.microsoft.com/en-us/python/api/overview/azure/ml/?view=azure-ml-py) guide.

In this [source](./Build_Operate/experiments_with_scripts/01-Runing-experiments-on-azure-with-scripts.ipynb), I use Azure Machine Learning Python SDK to create an Experiment, configure and run a script that uses LogisticRegresion to predict diabetic status, and Registers the model output in an Azure Machine Learning workspace.

### Experiment job run

Here, the Experiment is created in Azure with `local compute target`. An Azure compute instance target can also be specified using the `RunConfiguration` of the `ScriptRunConfig`. "Test data size (`test_size`) and Regularization rate (`reg_rate`) for the `LogisticRegression` are passed as parameters. Other parameters such as a registered dataset in a data store could also be passed.


```python
from azureml.core import ScriptRunConfig, Environment

env = Environment.from_existing_conda_environment(name="automate", conda_environment_name= "automate")
env.python.user_managed_dependencies = True
env.register(workspace=ws)

# Define arguments / parameters

test_size = 0.30
reg_rate = 0.01

script_config = ScriptRunConfig(
    source_directory=".",
    script="diabetes-training.py",
    arguments=["--reg-rate", reg_rate, "--test-size", test_size],
    environment=env,
)

run = experiment.submit(config=script_config)
run.wait_for_completion(show_output=False)
```

<img src="./Build_Operate/experiments_with_scripts/1. job run.png" alt="drawing" width="1200"/>


### Registered model

<img src="./Build_Operate/experiments_with_scripts/2. registered model.png" alt="drawing" width="1200"/>
