import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from IPython.display import Image                 # for displaying an image via code
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error


### --------- Functions --------- ###
def create_features(df, label=None):
    """
    Creates time series features from datetime index
    """
    df["date"] = df.index
    df["hour"] = df["date"].dt.hour
    df["dayofweek"] = df["date"].dt.dayofweek
    df["quarter"] = df["date"].dt.quarter
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["dayofyear"] = df["date"].dt.dayofyear
    df["dayofmonth"] = df["date"].dt.day
    df["weekofyear"] = df["date"].dt.weekofyear
    df["pjme_6_hrs_lag"] = df["PJME_MW"].shift(6)
    df["pjme_12_hrs_lag"] = df["PJME_MW"].shift(12)
    df["pjme_24_hrs_lag"] = df["PJME_MW"].shift(24)
    df["pjme_6_hrs_mean"] = df["PJME_MW"].rolling(window = 6).mean()
    df["pjme_12_hrs_mean"] = df["PJME_MW"].rolling(window = 12).mean()
    df["pjme_24_hrs_mean"] = df["PJME_MW"].rolling(window = 24).mean()
    df["pjme_6_hrs_std"] = df["PJME_MW"].rolling(window = 6).std()
    df["pjme_12_hrs_std"] = df["PJME_MW"].rolling(window = 12).std()
    df["pjme_24_hrs_std"] = df["PJME_MW"].rolling(window = 24).std()
    df["pjme_6_hrs_max"] = df["PJME_MW"].rolling(window = 6).max()
    df["pjme_12_hrs_max"] = df["PJME_MW"].rolling(window = 12).max()
    df["pjme_24_hrs_max"] = df["PJME_MW"].rolling(window = 24).max()
    df["pjme_6_hrs_min"] = df["PJME_MW"].rolling(window = 6).min()
    df["pjme_12_hrs_min"] = df["PJME_MW"].rolling(window = 12).min()
    df["pjme_24_hrs_min"] = df["PJME_MW"].rolling(window = 24).min()
    
    df_with_features = df[["hour","dayofweek","quarter","month","year","dayofyear","dayofmonth",
            "weekofyear","pjme_6_hrs_lag","pjme_24_hrs_lag","pjme_6_hrs_mean",
            "pjme_12_hrs_mean","pjme_24_hrs_mean","pjme_6_hrs_std","pjme_12_hrs_std",
            "pjme_24_hrs_std","pjme_6_hrs_max","pjme_12_hrs_max","pjme_24_hrs_max",
            "pjme_6_hrs_min","pjme_12_hrs_min","pjme_24_hrs_min"]]
    
    if label:
        y = df[label]
        return df_with_features, y
    return df_with_features


def mean_absolute_percentage_error(y_true, y_pred): 
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


### --------- Graphing Setup --------- ###

plt.style.use("fivethirtyeight")
color_pal = ["#F8766D", "#D39200", "#93AA00", "#00BA38", "#00C19F", "#00B9E3", "#619CFF", "#DB72FB"]


### --------- Main --------- ###

# read data
pjme = pd.read_csv("Data/PJME_hourly.csv", index_col=[0], parse_dates=[0])
print(pjme.head())

# chart1
# _ = pjme.plot(style=".", figsize=(15,5), color=color_pal[0], title="PJM East")
# Image(filename="")



# Train/Test Split
split_date = "01-Jan-2015"
pjme_train = pjme.loc[pjme.index <= split_date].copy()
pjme_test = pjme.loc[pjme.index > split_date].copy()

# chart2
# _ = pjme_test \
#     .rename(columns={"PJME_MW": "TEST SET"}) \
#     .join(pjme_train.rename(columns={"PJME_MW": "TRAINING SET"}), how="outer") \
#     .plot(figsize=(15,5), title="PJM East", style=".")



# Rolling and Lag Features
# Lag features are added to convert time series forecasting as a supervised Machine 
# Learning Problem.
# - Lags can be thought as we are using past information to forecast future 
# Rolling Window statistic are added, typical rolling window statistic includes mean, 
# median, std deviation over a fixed window sized statistic

# print(pjme_train.head(7))

# print(pjme_train.head(7).shift(1))

# Rolling window consists of fixed size windows over which statistics like mean, median, 
# mode, min, max, std deviation etc 

# print(pjme_train.head(7))

# rolling mean with window size = 2
# print(pjme_train.head(7).rolling(window = 2).mean())


# Create features
X_train, y_train = create_features(pjme_train, label='PJME_MW')
X_test, y_test = create_features(pjme_test, label='PJME_MW')

# create xgboost model
reg = xgb.XGBRegressor(n_estimators=1000)
reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds=50,
        verbose=False)


# visualize the feature importances
# chart3
# _ = plot_importance(reg, height=0.9 ,max_num_features = 10)

# forecast on test set
pjme_test['MW_Prediction'] = reg.predict(X_test)
pjme_all = pd.concat([pjme_test, pjme_train], sort=False)

# chart4
# _ = pjme_all[['PJME_MW','MW_Prediction']].plot(figsize=(15, 5))

# Look at first month of predictions
# Plot a daily forecast with the actuals
# chart5
# f, ax = plt.subplots(1)
# f.set_figheight(5)
# f.set_figwidth(15)
# _ = pjme_all[['MW_Prediction','PJME_MW']].plot(ax=ax,style=['-','.'])
# ax.set_xbound(lower='01-01-2015', upper='02-01-2015')
# ax.set_ylim(0, 60000)
# plot = plt.suptitle('January 2015 Forecast vs Actuals')


# Plot a weekly forecast with the actuals
# chart6
# f, ax = plt.subplots(1)
# f.set_figheight(5)
# f.set_figwidth(15)
# _ = pjme_all[['MW_Prediction','PJME_MW']].plot(ax=ax,style=['-','.'])
# ax.set_xbound(lower='01-01-2015', upper='01-08-2015')
# ax.set_ylim(0, 60000)
# plot = plt.suptitle('First Week of January Forecast vs Actuals')

# chart7
# f, ax = plt.subplots(1)
# f.set_figheight(5)
# f.set_figwidth(15)
# _ = pjme_all[['MW_Prediction','PJME_MW']].plot(ax=ax,style=['-','.'])
# ax.set_ylim(0, 60000)
# ax.set_xbound(lower='07-01-2015', upper='07-08-2015')
# plot = plt.suptitle('First Week of July Forecast vs Actuals')


# Error Metrics On Test Set
mse = mean_squared_error(y_true=pjme_test['PJME_MW'],
                            y_pred=pjme_test['MW_Prediction'])
mae = mean_absolute_error(y_true=pjme_test['PJME_MW'],
                    y_pred=pjme_test['MW_Prediction'])
# mean absolute percent error gives an easy to interperate percentage showing how off the 
# predictions are. MAPE isn't included in sklearn so we need to use a custom function
mape = mean_absolute_percentage_error(y_true=pjme_test['PJME_MW'],
                                        y_pred=pjme_test['MW_Prediction'])

# print(f"mean_squared_error: {mse}")
# print(f"mean_absolute_error: {mae}")
# print(f"mean_absolute_percentage_error: {mape}")


