# Plot distrubtions step

import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
from itertools import combinations

# Create a function that we can re-use
def plot_correlations(data):
    """
    This function will make a correlation graph and save it
    """
    correlation = data.corr()
    print("Correlation between features\n", correlation)

    fig = plt.figure(figsize=(10, 12))
    sns.heatmap(data=correlation, annot=True)
    plt.title("Correlation betweeen features")

    # Save plot
    filename = "outputs/correlations-between-features.png"
    os.makedirs("outputs", exist_ok=True)
    fig.savefig(filename)


def plot_distribution(var_data, column_name=None):
    """
    This function will make a distribution (graph) and save it
    """

    # Get statistics
    min_val = var_data.min()
    max_val = var_data.max()
    mean_val = var_data.mean()
    med_val = var_data.median()
    mod_val = var_data.mode()[0]

    print(
        "{} Statistics:\nMinimum:{:.2f}\nMean:{:.2f}\nMedian:{:.2f}\nMode:{:.2f}\nMaximum:{:.2f}\n".format(
            "" if column_name is None else column_name,
            min_val,
            mean_val,
            med_val,
            mod_val,
            max_val,
        )
    )

    # Create a figure for 2 subplots (2 rows, 1 column)
    fig, ax = plt.subplots(2, 1, figsize=(10, 4))

    # Plot the histogram
    ax[0].hist(var_data)
    ax[0].set_ylabel("Frequency")

    # Add lines for the mean, median, and mode
    ax[0].axvline(x=min_val, color="gray", linestyle="dashed", linewidth=2)
    ax[0].axvline(x=mean_val, color="cyan", linestyle="dashed", linewidth=2)
    ax[0].axvline(x=med_val, color="red", linestyle="dashed", linewidth=2)
    ax[0].axvline(x=mod_val, color="yellow", linestyle="dashed", linewidth=2)
    ax[0].axvline(x=max_val, color="gray", linestyle="dashed", linewidth=2)
    ax[0].legend()

    # Plot the boxplot
    ax[1].boxplot(var_data, vert=False)
    xlabel = "Value" if column_name is None else column_name
    ax[1].set_xlabel(xlabel)

    # Add a title to the Figure
    title = (
        "Data Distribution"
        if column_name is None
        else "{} Data Distribution".format(column_name)
    )
    fig.suptitle(title)

    # Save plot
    filename = "outputs/{}-distribution.png".format(column_name)
    os.makedirs("outputs", exist_ok=True)
    fig.savefig(filename)

def plot_scatters(x_y_data):
    """
    Plot scatter plots with :y_column: on y-axis and save them. 
    """
    
    x_column = x_y_data.columns.values[0]
    y_column = x_y_data.columns.values[1]

    fig = plt.figure(figsize=(10, 12))
    sns.regplot(data=x_y_data,x=x_column, y=y_column)
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title("Scatter plot of {} vs {}".format(x_column,y_column))

    # Save plot
    filename = "outputs/Scatter plot of {} vs {}.png".format(x_column,y_column)
    os.makedirs("outputs", exist_ok=True)
    fig.savefig(filename)

print("Loading Data...")
diabetes = pd.read_csv("../../data/diabetes.csv")

# plot correlations
plot_correlations(data=diabetes)

# plot distributions
columns = diabetes.columns.values
for x in columns:
    plot_distribution(var_data=diabetes[x],column_name=x)

# plot scatter plots
columns = set(columns)
exlude_column = set(["Diabetic", "PatientID"])

column_comb=list(combinations(columns-exlude_column,2))
column_comb = [list(x) for x in column_comb]

for x_y_pairs in column_comb:
    plot_scatters(diabetes[x_y_pairs])

