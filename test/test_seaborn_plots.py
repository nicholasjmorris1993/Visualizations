import numpy as np
import pandas as pd
from xgboost.sklearn import XGBClassifier, XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
import sys
sys.path.append("/home/nick/Visualizations/src")
from seaborn_plots import (
    matrix_plot,
    parity_plot,
    pairs_plot,
    series_plot,
    scatter_plot,
    corr_plot,
)


data = pd.read_csv("/home/nick/Visualizations/test/LungCap.csv")
df = pd.read_csv("/home/nick/Visualizations/test/traffic.txt", sep="\t")

def tests(data, df):
    testing = Testing()
    testing.test_matrix_plot(data)  # predict smoking status based on lung capacity and other variables
    testing.test_parity_plot(data)  # predict lung capacity based on smoking status and other variables
    testing.test_pairs_plot(data)  # plot pairwise scatterplots of lung capacity
    testing.test_series_plot(df)  # plot traffic alongside traffic with noise
    testing.test_scatter_plot(data)  # plot lung capacity v. age
    testing.test_corr_plot(data)  # plot lung capacity correlations


class Testing:
    def test_matrix_plot(self, data):
        data = data.copy().sample(frac=1, random_state=0).reset_index(drop=True)  # shuffle the data
        train = data.copy().head(int(len(data)*0.5))
        test = data.copy().tail(int(len(data)*0.5))

        X = train.copy().drop(columns=["Smoke no", "Smoke yes"])
        Y = train.copy()[["Smoke yes"]]

        classes_weights = class_weight.compute_sample_weight(
            class_weight="balanced",
            y=Y["Smoke yes"]
        )

        model = XGBClassifier(
            booster="gbtree",
            n_estimators=100, 
            learning_rate=0.1,
            max_depth=7,
            min_child_weight=1,
            colsample_bytree=0.8,
            subsample=0.8,
            random_state=42,
        )
        model.fit(X, Y, sample_weight=classes_weights)

        X = test.copy().drop(columns=["Smoke no", "Smoke yes"])
        Y = test.copy()[["Smoke yes"]]

        y_pred = model.predict(X)
        y_true = Y.to_numpy().ravel().astype("int")

        cmatrix = confusion_matrix(y_true, y_pred)
        matrix_plot(cmatrix, title="Smoking Status", save=True)

    def test_parity_plot(self, data):
        data = data.copy().sample(frac=1, random_state=0).reset_index(drop=True)  # shuffle the data
        train = data.copy().head(int(len(data)*0.5))
        test = data.copy().tail(int(len(data)*0.5))

        X = train.copy().drop(columns="LungCap")
        Y = train.copy()[["LungCap"]]

        model = XGBRegressor(
            booster="gbtree",
            n_estimators=100, 
            learning_rate=0.1,
            max_depth=7,
            min_child_weight=1,
            colsample_bytree=0.8,
            subsample=0.8,
            random_state=42,
        )
        model.fit(X, Y)

        X = test.copy().drop(columns="LungCap")
        Y = test.copy()[["LungCap"]]

        y_pred = model.predict(X)
        y_true = Y.to_numpy().ravel()

        metric = mean_squared_error(
            y_true=y_true, 
            y_pred=y_pred, 
            squared=False,
        )
        metric = f"RMSE: {round(metric, 6)}"

        parity_plot(
            predict=y_pred, 
            actual=y_true, 
            title=f"Lung Capacity\n{metric}", 
            save=True,
        )

    def test_pairs_plot(self, data):
        pairs_plot(
            data, 
            vars=data.columns[:3].tolist(), 
            color="LungCap", 
            title="Lung Capacity Dataset", 
            save=True,
        )

    def test_series_plot(self, df):
        np.random.seed(0)
        df["Noise"] = df["Vehicles"] * np.random.triangular(
            left=0.8, 
            mode=1, 
            right=1.2, 
            size=df.shape[0],
        )

        series_plot(
            df["Noise"], 
            df["Vehicles"], 
            title="Traffic With Noise", 
            save=True,
        )

    def test_scatter_plot(self, data):
        scatter_plot(
            data, 
            x="Age", 
            y="LungCap", 
            color="Gender male", 
            title="Lung Capacity v. Age", 
            save=True,
        )

    def test_corr_plot(self, data):
        corr_plot(data, title="Lung Capacity Correlations", save=True)

# test each seaborn plot
tests(data, df)
