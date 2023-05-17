import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit              # for Time Series Cross Validation


# read in data and store as a dataframe
df = pd.read_csv('Data/PJME_hourly.csv')
# set Datetime column as the index and format as a datetime object
df = df.set_index('Datetime')
df.index = pd.to_datetime(df.index)


# 2017-01-28,5881
target_map = df['PJME_MW'].to_dict()
print(target_map)
df['lag1'] = (df.index - pd.Timedelta('364 days')).map(target_map)
# print(df.tail())
#                      PJME_MW     lag1
# Datetime
# 2018-01-01 20:00:00  44284.0  34600.0
# 2018-01-01 21:00:00  43751.0  33674.0
# 2018-01-01 22:00:00  42402.0  32035.0
# 2018-01-01 23:00:00  40164.0  29870.0
# 2018-01-02 00:00:00  38608.0  27771.0


# print(44284.0-5881)
# 38403.0











