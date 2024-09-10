
# Machine Learning with Microsoft Azure

**Microsoft Azure** provides the cloud infrastructure for scalable, production-grade Machine Learning with minimal code. Azure offers tools for the full lifecycle of Machine Learning, including the productionization and monitoring of models.

In previous Machine Learning work [[1](https://github.com/kennedyopokuasare/IBM_datascience), [2](https://github.com/kennedyopokuasare), [3](https://github.com/kennedyopokuasare/carat-analysis)], I have executed entire machine learning pipelines, from data cleaning to prediction, using my personal computer, Jupyter Notebook, and Snakemake for workflow management. There are bottlenecks when training machine learning models with large datasets on a personal computer with limited resources, especially when multiple experiments need to be conducted — a case in point is when I trained [five machine learning algorithms to predict the depression status of participants](https://www.sciencedirect.com/science/article/pii/S1574119222000566). 

This repository presents a project where I use Microsoft Azure Machine Learning Cloud services for training, inferencing and Productionization of Machine Learning Models. 

## Exploratory Data Analysis

In `Azure ML Studio`, notebooks can be run just like you would on your computer. First, you need to provision a `compute instance` that will run the computations in the notebook.

For instance:
* I explored the student grades and hours of study dataset and examined the distribution of the students' grades. [Source](./With_code/02-visualize-data.ipynb)
* I used the K-Means clustering algorithm to cluster the seeds dataset based on two principal components of the dataset. [Source](./With_code/01-clustering-introduction.ipynb)


## Automated Machine Learning in Azure ML studio

## AutoML completed, with MaxAbsScaler and XGBoostRegressor

<img src="./AutoML/1.%20AutoML%20completed.png" alt="drawing" width="1200"/>

### Explaining the AutoML regression model 

<img src="./AutoML/2. Explanations and feature importance.png" alt="drawing" width="1200"/>