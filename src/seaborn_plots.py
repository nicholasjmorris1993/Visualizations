# -*- coding: utf-8 -*-
"""
Creates plots for analyzing model performance

@author: Nick
"""


import re
import numpy as np
import scipy.cluster.hierarchy as sch
import seaborn as sns
import matplotlib.pyplot as plt


def matrix_plot(matrix, title="Matrix Plot", save=False):
    # set up labels for the plot
    group_names = ["True Neg","False Pos","False Neg","True Pos"]
    group_counts = ["{0:0.0f}".format(value) for value in
                    matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                         matrix.flatten()/np.sum(matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    # plot the predictions
    fig, ax = plt.subplots()
    sns.heatmap(matrix, annot=labels, fmt="", cmap="Blues", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predict")
    ax.set_ylabel("Actual")
    if save:
        title = re.sub("[^A-Za-z0-9]+", "", title)
        plt.savefig(title + ".png")
    else:
        plt.show()

def parity_plot(predict, actual, title="Parity Plot", alpha=2/3, save=False):
    # plot the predictions
    fig, ax = plt.subplots()
    sns.scatterplot(x=actual, y=predict, color="blue", alpha=alpha, ax=ax)
    sns.lineplot(x=actual, y=actual, color="red", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Predict")
    ax.set_xlabel("Actual")
    if save:
        title = re.sub("[^A-Za-z0-9]+", "", title)
        plt.savefig(title + ".png")
    else:
        plt.show()

def pairs_plot(data, vars, color, title="Pairs Plot", save=False):
    p = sns.pairplot(data, vars=vars, hue=color)
    p.fig.suptitle(title, y=1.08)
    if save:
        title = re.sub("[^A-Za-z0-9]+", "", title)
        plt.savefig(title + ".png")
    else:
        plt.show()

def series_plot(predict, actual, title="Series Plot", save=False):
    # plot the predictions
    fig, ax = plt.subplots()
    idx = [i for i in range(len(predict))]
    sns.lineplot(x=idx, y=predict, color="blue", ax=ax)
    sns.lineplot(x=idx, y=actual, color="red", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    if save:
        title = re.sub("[^A-Za-z0-9]+", "", title)
        plt.savefig(title + ".png")
    else:
        plt.show()

def scatter_plot(data, x, y, color, title="Scatter Plot", legend=True, save=False):
    p = sns.lmplot(x=x, y=y, data=data, fit_reg=False, hue=color, 
                   legend=legend)
    p.fig.suptitle(title, y=1.08)
    if save:
        title = re.sub("[^A-Za-z0-9]+", "", title)
        plt.savefig(title + ".png")
    else:
        plt.show()


def corr_plot(df, size=10, method="ward", title="Correlation Plot", save=False):
    # group columns together with hierarchical clustering
    X = df.corr().values
    d = sch.distance.pdist(X)
    L = sch.linkage(d, method=method)
    ind = sch.fcluster(L, 0.5*d.max(), "distance")
    columns = [df.columns.tolist()[i] for i in list((np.argsort(ind)))]
    df = df.reindex(columns, axis=1)
    
    # compute the correlation matrix for the received dataframe
    corr = df.corr()
    
    # plot the correlation matrix
    fig, ax = plt.subplots(figsize=(size, size))
    cax = ax.matshow(corr, cmap="RdYlGn")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90);
    plt.yticks(range(len(corr.columns)), corr.columns);
    
    # add the colorbar legend
    fig.colorbar(cax, ticks=[-1, 0, 1], aspect=40, shrink=.8)

    fig.suptitle(title, y=1.08)
    if save:
        title = re.sub("[^A-Za-z0-9]+", "", title)
        plt.savefig(title + ".png")
    else:
        plt.show()

