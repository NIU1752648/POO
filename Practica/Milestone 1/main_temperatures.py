import pandas as pd
import numpy as np

from RandomForestEvaluator import RandomForestEvaluator, RandomForestRegressor

def load_daily_min_temperatures():
    df = pd.read_csv('daily-min-temperatures.csv')
    day = pd.DatetimeIndex(df.Date).day.to_numpy()  # 1...31
    month = pd.DatetimeIndex(df.Date).month.to_numpy()  # 1...12
    year = pd.DatetimeIndex(df.Date).year.to_numpy()  # 1981...1999
    X = np.vstack([day, month, year]).T  # np array of 3 columns
    y = df.Temp.to_numpy()
    X = X[~np.isnan(y)]
    y = y[~np.isnan(y)]
    return X, y

if __name__ == "__main__":
    X, y = load_daily_min_temperatures()
    random_regressor_temps = RandomForestEvaluator(
        "Daily Min Temperatures",
        RandomForestRegressor(),
        X, y
    )

    random_regressor_temps.train()

    random_regressor_temps.test_regression()

    #random_regressor_temps.print_trees()
    random_regressor_temps.plot_fi()
