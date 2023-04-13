import numpy as np
import pandas as pd
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error
import sys
sys.path.append("/home/nick/Visualizations/src")
from plotly_plots import (
    cormap,
    scatter,
    line,
    parity,
    series,
    histogram,
    pairs,
)


data = pd.read_csv("/home/nick/Visualizations/test/LungCap.csv")
df = pd.read_csv("/home/nick/Visualizations/test/traffic.txt", sep="\t")

def tests(data, df):
    testing = Testing()
    testing.test_cormap(data)  # plot the lung capacity correlations
    testing.test_scatter(data)  # plot lung capacity v. age
    testing.test_line(df)  # plot traffic over time
    testing.test_parity(data)  # predict lung capacity based on smoking status and other variables
    testing.test_series(df)  # plot traffic alongside traffic with noise
    testing.test_histogram(data)  # plot lung capacity
    testing.test_pairs(data)  # plot pairwise scatterplots of lung capacity


class Testing:
    def test_cormap(self, data):
        cormap(data, title="Lung Capacity Correlations", font_size=16)

    def test_scatter(self, data):
        scatter(
            data, 
            x="Age", 
            y="LungCap", 
            color="Gender male", 
            title="Lung Capacity v. Age", 
            font_size=16,
        )
    
    def test_line(self, df):
        line(
            df, 
            x="Day", 
            y="Vehicles", 
            color=None, 
            title="Traffic", 
            font_size=16,
        )

    def test_parity(self, data):
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
        predictions = pd.DataFrame({
            "Predicted": y_pred,
            "Actual": y_true,
        })

        metric = mean_squared_error(
            y_true=y_true, 
            y_pred=y_pred, 
            squared=False,
        )
        metric = f"RMSE: {round(metric, 6)}"

        parity(
            predictions,
            predict="Predicted", 
            actual="Actual", 
            title=f"Lung Capacity {metric}", 
            font_size=16,
        )

    def test_series(self, df):
        np.random.seed(0)
        df["Noise"] = df["Vehicles"] * np.random.triangular(
            left=0.8, 
            mode=1, 
            right=1.2, 
            size=df.shape[0],
        )

        series(
            df,
            predict="Noise", 
            actual="Vehicles", 
            title="Traffic With Noise", 
            font_size=16,
        )

    def test_histogram(self, data):
        histogram(
            data,
            x="LungCap",
            title="Lung Capacity",
            font_size=16,
        )

    def test_pairs(self, data):
        pairs(
            data.iloc[:,:3],
            title="Lung Capacity Dataset",
            font_size=16,
        )


tests(data, df)
