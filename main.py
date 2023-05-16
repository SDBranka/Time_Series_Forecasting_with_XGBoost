import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error


### --------- Functions --------- ###
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


### --------- Main --------- ###
# set up plot colors and style
color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')


# read in data and store as a dataframe
df = pd.read_csv('Data/PJME_hourly.csv')
# print(df.head())
#               Datetime  PJME_MW
# 0  2002-12-31 01:00:00  26498.0
# 1  2002-12-31 02:00:00  25147.0
# 2  2002-12-31 03:00:00  24574.0
# 3  2002-12-31 04:00:00  24393.0
# 4  2002-12-31 05:00:00  24860.0


# set Datetime column as the index and format as a datetime object
df = df.set_index('Datetime')
df.index = pd.to_datetime(df.index)
# print(df.head())
#                      PJME_MW
# Datetime
# 2002-12-31 01:00:00  26498.0
# 2002-12-31 02:00:00  25147.0
# 2002-12-31 03:00:00  24574.0
# 2002-12-31 04:00:00  24393.0
# 2002-12-31 05:00:00  24860.0

# chart1
# df.plot(style='.',
#         figsize=(15, 5),
#         color=color_pal[0],
#         title='PJME Energy Use in MW')
# plt.show()


# Split data into testing and training sets
train = df.loc[df.index < '01-01-2015']
test = df.loc[df.index >= '01-01-2015']

# chart2
# fig, ax = plt.subplots(figsize=(15, 5))
# train.plot(ax=ax, label='Training Set', title='Chart 2 Data Train/Test Split')
# test.plot(ax=ax, label='Test Set')
# ax.axvline('01-01-2015', color='black', ls='--')
# ax.legend(['Training Set', 'Test Set'])
# plt.show()

# chart3
# df.loc[(df.index > '01-01-2010') & (df.index < '01-08-2010')] \
#     .plot(figsize=(15, 5), title='Chart 3 Week Of Data')
# plt.show()


# create feature
df = create_features(df)


# Visualize the Feature/Target(Label) Relationship
# chart4
# fig, ax = plt.subplots(figsize=(10, 8))
# sns.boxplot(data=df, x='hour', y='PJME_MW')
# ax.set_title('Chart 4 MW by Hour')
# plt.show()

# chart5
# fig, ax = plt.subplots(figsize=(10, 8))
# sns.boxplot(data=df, x='month', y='PJME_MW', palette='Blues')
# ax.set_title('Chart 5 MW by Month')
# plt.show()


# Create the model
train = create_features(train)
test = create_features(test)

FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']
TARGET = 'PJME_MW'

X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',    
                        n_estimators=1000,
                        early_stopping_rounds=50,
                        objective='reg:linear',
                        max_depth=3,
                        learning_rate=0.01
                        )
reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100)


# Feature Importance
# chart6
# fi = pd.DataFrame(data=reg.feature_importances_,
#                     index=reg.feature_names_in_,
#                     columns=['importance']
#                     )
# fi.sort_values('importance').plot(kind='barh', title='Chart 6 Feature Importance')
# plt.show()


# Forecast on Test
test['prediction'] = reg.predict(X_test)
df = df.merge(test[['prediction']], how='left', left_index=True, right_index=True)

# chart7
# ax = df[['PJME_MW']].plot(figsize=(15, 5))
# df['prediction'].plot(ax=ax, style='.')
# plt.legend(['Actual Data', 'Predictions'])
# ax.set_title('Chart 7 Raw Dat and Prediction')
# plt.show()

# chart8
# ax = df.loc[(df.index > '04-01-2018') & (df.index < '04-08-2018')]['PJME_MW'] \
#     .plot(figsize=(15, 5), title='Chart 8 Week Of Data')
# df.loc[(df.index > '04-01-2018') & (df.index < '04-08-2018')]['prediction'] \
#     .plot(style='.')
# plt.legend(['Actual Data','Prediction'])
# plt.show()


# Score (RMSE)
score = np.sqrt(mean_squared_error(test['PJME_MW'], test['prediction']))
# print(f'RMSE Score on Test set: {score:0.2f}')
# RMSE Score on Test set: 3721.75


# Calculate Error
# Look at the worst and best predicted days
test['error'] = np.abs(test[TARGET] - test['prediction'])
test['date'] = test.index.date
# print(test.groupby(['date'])['error'].mean().sort_values(ascending=False).head(10))
# date
# 2016-08-13    12839.597087
# 2016-08-14    12780.209961
# 2016-09-10    11356.302979
# 2015-02-20    10965.982259
# 2016-09-09    10864.954834
# 2018-01-06    10506.845622
# 2016-08-12    10124.051595
# 2015-02-21     9881.803711
# 2015-02-16     9781.552246
# 2018-01-07     9739.144206
# Name: error, dtype: float64





















