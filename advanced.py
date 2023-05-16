import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit              # for Time Series Cross Validation


### --------- Functions --------- #
# Create time series features based on time series index.
def create_features(df):
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df


def add_lags(df):
    target_map = df['PJME_MW'].to_dict()
    df['lag1'] = (df.index - pd.Timedelta('364 days')).map(target_map)
    df['lag2'] = (df.index - pd.Timedelta('728 days')).map(target_map)
    df['lag3'] = (df.index - pd.Timedelta('1092 days')).map(target_map)
    return df

### --------- Main --------- ###
# set up plot colors and style
color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')


# read in data and store as a dataframe
df = pd.read_csv('Data/PJME_hourly.csv')
# set Datetime column as the index and format as a datetime object
df = df.set_index('Datetime')
df.index = pd.to_datetime(df.index)

# chart1
# df.plot(style='.',
#         figsize=(15, 5),
#         color=color_pal[0],
#         title='PJME Energy Use in MW')
# plt.show()


# Outlier Analysis and removal
# chart2
# df['PJME_MW'].plot(kind='hist', bins=500,
#                     title="Chart 2 MW Bins Hist"
#                     )
# plt.show()

# chart3
# df.query('PJME_MW < 19_000')['PJME_MW'].plot(style='.',
#                                             figsize=(15, 5),
#                                             color=color_pal[5],
#                                             title='Chart 3 Outliers'
#                                             )
# plt.show()

# 
df = df.query('PJME_MW > 19_000').copy()
# print(df.head())
#                      PJME_MW
# Datetime
# 2002-12-31 01:00:00  26498.0
# 2002-12-31 02:00:00  25147.0
# 2002-12-31 03:00:00  24574.0
# 2002-12-31 04:00:00  24393.0
# 2002-12-31 05:00:00  24860.0



# Reviewing: Train / Test Split
# Split the data into training and testing sets
train = df.loc[df.index < '01-01-2015']
test = df.loc[df.index >= '01-01-2015']

# chart4
# fig, ax = plt.subplots(figsize=(15, 5))
# train.plot(ax=ax, label='Training Set', title='Chart 4 Data Train/Test Split')
# test.plot(ax=ax, label='Test Set')
# ax.axvline('01-01-2015', color='black', ls='--')
# ax.legend(['Training Set', 'Test Set'])
# plt.show()


# Time Series Cross Validation
tss = TimeSeriesSplit(n_splits=5, test_size=24*365*1, gap=24)
# print(tss)
# TimeSeriesSplit(gap=24, max_train_size=None, n_splits=5, test_size=8760)
df = df.sort_index()
# print(df.head())
# Datetime
# 2002-01-01 01:00:00  30393.0
# 2002-01-01 02:00:00  29265.0
# 2002-01-01 03:00:00  28357.0
# 2002-01-01 04:00:00  27899.0
# 2002-01-01 05:00:00  28057.0


# chart5
# fig, axs = plt.subplots(5, 1, figsize=(15, 15), sharex=True)

# fold = 0
# for train_idx, val_idx in tss.split(df):
#     train = df.iloc[train_idx]
#     test = df.iloc[val_idx]
#     train['PJME_MW'].plot(ax=axs[fold],
#                             label='Training Set',
#                             title=f'Chart 5 Data Train/Test Split Fold {fold}')
#     test['PJME_MW'].plot(ax=axs[fold],
#                             label='Test Set')
#     axs[fold].axvline(test.index.min(), color='black', ls='--')
#     fold += 1
# plt.show()


# Forecasting Horizon Explained
# The forecast horizon is the length of time into the future for which 
# forecasts are to be prepared. These generally vary from short-term 
# forecasting horizons (less than three months) to long-term horizons 
# (more than two years).

# create features
df = create_features(df)
# print(df.head())
#                      PJME_MW  hour  ...  dayofmonth  weekofyear
# Datetime                            ...
# 2002-01-01 01:00:00  30393.0     1  ...           1           1
# 2002-01-01 02:00:00  29265.0     2  ...           1           1
# 2002-01-01 03:00:00  28357.0     3  ...           1           1
# 2002-01-01 04:00:00  27899.0     4  ...           1           1
# 2002-01-01 05:00:00  28057.0     5  ...           1           1

# [5 rows x 9 columns]


# Lag Features
# What was the target (x) days in the past
df = add_lags(df)
# print(df.head())
#                      PJME_MW  hour  dayofweek  ...  lag1  lag2  lag3
# Datetime                                       ...
# 2002-01-01 01:00:00  30393.0     1          1  ...   NaN   NaN   NaN
# 2002-01-01 02:00:00  29265.0     2          1  ...   NaN   NaN   NaN
# 2002-01-01 03:00:00  28357.0     3          1  ...   NaN   NaN   NaN
# 2002-01-01 04:00:00  27899.0     4          1  ...   NaN   NaN   NaN
# 2002-01-01 05:00:00  28057.0     5          1  ...   NaN   NaN   NaN

# [5 rows x 12 columns]


# Train Using Cross Validation
tss = TimeSeriesSplit(n_splits=5, test_size=24*365*1, gap=24)
df = df.sort_index()
# print(df.head())
#                      PJME_MW  hour  dayofweek  ...  lag1  lag2  lag3
# Datetime                                       ...
# 2002-01-01 01:00:00  30393.0     1          1  ...   NaN   NaN   NaN
# 2002-01-01 02:00:00  29265.0     2          1  ...   NaN   NaN   NaN
# 2002-01-01 03:00:00  28357.0     3          1  ...   NaN   NaN   NaN
# 2002-01-01 04:00:00  27899.0     4          1  ...   NaN   NaN   NaN
# 2002-01-01 05:00:00  28057.0     5          1  ...   NaN   NaN   NaN

# [5 rows x 12 columns]


fold = 0
preds = []
scores = []
for train_idx, val_idx in tss.split(df):
    train = df.iloc[train_idx]
    test = df.iloc[val_idx]

    train = create_features(train)
    test = create_features(test)

FEATURES = ['dayofyear', 'hour', 'dayofweek', 
            'quarter', 'month','year',
            'lag1','lag2','lag3'
            ]
TARGET = 'PJME_MW'

X_train = train[FEATURES]
# print(X_train)
#                      dayofyear  hour  dayofweek  ...     lag1     lag2     lag3
# Datetime                                         ...
# 2002-01-01 01:00:00          1     1          1  ...      NaN      NaN      NaN
# 2002-01-01 02:00:00          1     2          1  ...      NaN      NaN      NaN
# 2002-01-01 03:00:00          1     3          1  ...      NaN      NaN      NaN
# 2002-01-01 04:00:00          1     4          1  ...      NaN      NaN      NaN
# 2002-01-01 05:00:00          1     5          1  ...      NaN      NaN      NaN
# ...                        ...   ...        ...  ...      ...      ...      ...
# 2017-08-01 20:00:00        213    20          1  ...  41056.0  46225.0  43934.0
# 2017-08-01 21:00:00        213    21          1  ...  40151.0  44510.0  42848.0
# 2017-08-01 22:00:00        213    22          1  ...  38662.0  42467.0  40861.0
# 2017-08-01 23:00:00        213    23          1  ...  35583.0  38646.0  37361.0
# 2017-08-02 00:00:00        214     0          2  ...  32181.0  34829.0  33743.0

# [136567 rows x 9 columns]

y_train = train[TARGET]
# print(y_train)
# Datetime
# 2002-01-01 01:00:00    30393.0
# 2002-01-01 02:00:00    29265.0
# 2002-01-01 03:00:00    28357.0
# 2002-01-01 04:00:00    27899.0
# 2002-01-01 05:00:00    28057.0
#                         ...
# 2017-08-01 20:00:00    45090.0
# 2017-08-01 21:00:00    43843.0
# 2017-08-01 22:00:00    41850.0
# 2017-08-01 23:00:00    38473.0
# 2017-08-02 00:00:00    35126.0
# Name: PJME_MW, Length: 136567, dtype: float64

X_test = test[FEATURES]
y_test = test[TARGET]

# create and fit the model
reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',    
                        n_estimators=1000,
                        early_stopping_rounds=50,
                        objective='reg:linear',
                        max_depth=3,
                        learning_rate=0.01
                        )
reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100
        )

y_pred = reg.predict(X_test)
preds.append(y_pred)
score = np.sqrt(mean_squared_error(y_test, y_pred))
scores.append(score)

print(f'Score across folds {np.mean(scores):0.4f}')
print(f'Fold scores:{scores}')


# Predicting the Future
# Retraining on all data
# To Predict the future we need an emtpy dataframe for future date ranges.
# Run those dates through our feature creation code + lag creation

# Retrain on all data
# df = create_features(df)

# FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year',
#             'lag1','lag2','lag3']
# TARGET = 'PJME_MW'

# X_all = df[FEATURES]
# y_all = df[TARGET]

# reg = xgb.XGBRegressor(base_score=0.5,
#                         booster='gbtree',    
#                         n_estimators=500,
#                         objective='reg:linear',
#                         max_depth=3,
#                         learning_rate=0.01
#                         )
# reg.fit(X_all, y_all,
#         eval_set=[(X_all, y_all)],
#         verbose=100
#         )































































