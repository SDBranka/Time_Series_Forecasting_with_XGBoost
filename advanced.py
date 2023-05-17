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
    # create a dictionary of all the label values from the dataset
    target_map = df['PJME_MW'].to_dict()
    # lag cannot be longer than the forecasting horizon
    # why 364 and not 365? This creates a number divisable by 7 so 
    # that we end up on the same day of the week so we don't have to 
    # worry about mapping out days of the week into the past
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
# Visually inspect the outlier values
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

# Filter out the outlier values
# create a df of the values that are greater than 19000 
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
# TimeSeriesSplit 
# n_splits - Number of splits. Must be at least 2
# test_size - Used to limit the size of the test set
# gap - Number of samples to exclude from the end of 
#       each train set before the test set.
# We plan to predict one year into the future with this model, so the
# test size is set to 24 hours * 365 days * 1 year
# We plan to exclude 24 hours between when the training set ends and
# the test set begins, so the gap is set to 24 (hours)
tss = TimeSeriesSplit(n_splits=5, test_size=24*365*1, gap=24)
# print(tss)
# TimeSeriesSplit(gap=24, max_train_size=None, n_splits=5, test_size=8760)

# the data frame is sorted by the datetime object index or else the
# TimeSeriesSplit will not work
df = df.sort_index()
# print(df.head())
# Datetime
# 2002-01-01 01:00:00  30393.0
# 2002-01-01 02:00:00  29265.0
# 2002-01-01 03:00:00  28357.0
# 2002-01-01 04:00:00  27899.0
# 2002-01-01 05:00:00  28057.0


# Display the TimeSeriesSplit to see how each fold works
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
# print(df.tail())
#                      PJME_MW  hour  dayofweek  ...     lag1     lag2     lag3
# Datetime                                       ...
# 2018-08-02 20:00:00  44057.0    20          3  ...  42256.0  41485.0  38804.0
# 2018-08-02 21:00:00  43256.0    21          3  ...  41210.0  40249.0  38748.0
# 2018-08-02 22:00:00  41552.0    22          3  ...  39525.0  38698.0  37330.0
# 2018-08-02 23:00:00  38500.0    23          3  ...  36490.0  35406.0  34552.0
# 2018-08-03 00:00:00  35486.0     0          4  ...  33539.0  32094.0  31695.0

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
# loop through the cross validation folds, train the model on each fold
# on each of the train/test splits, then score the models efficacy using
# the mean squared error, and save those scores to a list so that we can 
# evaluate the scores across all folds
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
    # print(y_pred)
    # [27884.035 27147.71  26344.05  ... 38405.37  36211.242 30370.074]

    preds.append(y_pred)
    # print(preds)
    # [array([27884.035, 27147.71 , 26344.05 , ..., 38405.37 , 36211.242,
    #        30370.074], dtype=float32)]

    score = np.sqrt(mean_squared_error(y_test, y_pred))
    # print(score)
    # 3996.298054855067

    scores.append(score)

# The more we tune the hyperparameters and the more features we add the
# better the scores across each fold should get

# print(f'Score across folds {np.mean(scores):0.4f}')
# print(f'Fold scores:{scores}')
# Score across folds 3750.6406
# Fold scores:[3753.2775219986684, 3434.3528874818867, 3475.9138463312997, 4093.36
# 08331481823, 3996.298054855067]


# Predicting the Future
# Retraining on all data
# To Predict the future we need an emtpy dataframe for future date ranges.
# Run those dates through our feature creation code + lag creation

# Retrain on all data
# before attempting to predict into the future we should train the model
# upon all available known data
# for this we will not be splitting the data into train/test sets

df = create_features(df)

FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year',
            'lag1','lag2','lag3']
TARGET = 'PJME_MW'

X_all = df[FEATURES]
y_all = df[TARGET]

# number of estimators changed to 500 in this process because looking at the 
# original training of the model it can be see that around the 500th epoch
# the model begins to overfit

# build the regressor
reg = xgb.XGBRegressor(base_score=0.5,
                        booster='gbtree',    
                        n_estimators=500,
                        objective='reg:linear',
                        max_depth=3,
                        learning_rate=0.01
                        )
reg.fit(X_all, y_all,
        eval_set=[(X_all, y_all)],
        verbose=100
        )


# Project into the future
# find the highest date in the df
# print(df.index.max())
# 2018-08-03 00:00:00


# create a range of dates from our last date until 1 month from that 
# date incremented by hour
future = pd.date_range('2018-08-03','2019-08-01', freq='1h')
# Create future dataframe making the future dates the index
future_df = pd.DataFrame(index=future)
# because we have lag dates, we will want to we will want to add this
# onto the existing dataframe
# create a column in each dataframe that will depict which dates occur
# in the future (True) and which are from existing data (False) 
future_df['isFuture'] = True
df['isFuture'] = False
# join the two dataframes to have one
df_and_future = pd.concat([df, future_df])
# create features and add lags
df_and_future = create_features(df_and_future)
df_and_future = add_lags(df_and_future)

# now that we have created our lag features we can extract out just 
# the data from dates that occur in the future
future_w_features = df_and_future.query('isFuture').copy()


# Predict into the future 1 year
future_w_features['pred'] = reg.predict(future_w_features[FEATURES])

# visualize the predictions
# chart6
# future_w_features['pred'].plot(figsize=(10, 5),
#                                 color=color_pal[4],
#                                 ms=1,
#                                 lw=1,
#                                 title='Chart 6 Future Predictions'
#                                 )
# plt.show()

# produce csv of the predictions
future_w_features.to_csv("predictions.csv")

# Saving Model For later
# Save model
reg.save_model('model.json')


reg_new = xgb.XGBRegressor()
reg_new.load_model('model.json')
future_w_features['pred'] = reg_new.predict(future_w_features[FEATURES])
# chart7
future_w_features['pred'].plot(figsize=(10, 5),
                                color=color_pal[4],
                                ms=1, lw=1,
                                title='Chart 7 Future Predictions From Saved Model'
                                )
plt.show()


