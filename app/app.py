# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt

# all scikit libraries
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge, ElasticNetCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# convert datetime object to float 
# (pandas dataset can only have one object type, and in this project, all columns will be floats)
def datetime_to_float(d):
    epoch = dt.datetime.utcfromtimestamp(0)
    total_seconds =  (d - epoch).total_seconds()
    # total_seconds will be in decimals (millisecond precision)
    return total_seconds
    
# import data from CSV
data = pd.read_csv('./data/ETHUSDT_1h.csv', parse_dates=['time'])
# print(data.columns.tolist())

# convert all datetime objects to floats
data['time'] = pd.to_datetime(data['time'], unit='s')
data['time'] = data['time'].apply(lambda x: datetime_to_float(x))
# print(data.head())
# print(data.info())

# plot graph of close prices
plt.figure(figsize=(15,8))
(data['close']).plot(color='blue', label='close')

# initialize legend, label x axis, and show graph
plt.legend()
plt.xlabel('time')
plt.show()




eth = data['close']

# calculate 200 day moving average
eth_ma = eth.rolling(window=200).mean()

# concatenate eth dataset with eth_ma dataset 
close = pd.concat([eth], axis=1)
close_ma = pd.concat([eth_ma], axis=1)
# close_ma.tail()

# plot moving average for closing price for cryptocurrencies
close_ma.plot(figsize=(15,8))
plt.title('5-Day Moving Average on Daily Closing Price')
plt.xlabel('time')
plt.ylabel('price in USD')
plt.show()




# calculate daily average price
data['daily_avg'] = (data['open'] + data['high'] + data['low'] + data['close']) / 4

# add column with daily avg a month prior
data['daily_avg_After_Month']=data['daily_avg'].shift(-168)



def rsi(df, periods = 14, ema = True):
    """
    Returns a pd.Series with the relative strength index.
    """
    close_delta = df['close'].diff()

    # Make two series: one for lower closes and one for higher closes
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    
    if ema == True:
	    # Use exponential moving average
        ma_up = up.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
        ma_down = down.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
    else:
        # Use simple moving average
        ma_up = up.rolling(window = periods, adjust=False).mean()
        ma_down = down.rolling(window = periods, adjust=False).mean()
        
    print("ma_up", ma_up, type(ma_up))
    rsi = ma_up / ma_down
    rsi = 100 - (100/(1 + rsi))
    return rsi

rsi_ETH = []

for i in range(14, 100):
    current_rsi = rsi(data.iloc[i-14:i])
    # print("current rsi: ", type(current_rsi))
    rsi_ETH.append(current_rsi)

# print(rsi_ETH[0:100])

data['rsi'] = rsi_ETH

# drop rows that contain empty values 
x_ETH = data.dropna().drop(['daily_avg_After_Month','daily_avg'], axis=1)
y_ETH = data.dropna()['daily_avg_After_Month']


# split data into train and test datasets
x_train_ETH, x_test_ETH, y_train_ETH, y_test_ETH = train_test_split(x_ETH, y_ETH, test_size=0.2, random_state=43)
x_forecast_ETH =  data.tail(168).drop(['daily_avg_After_Month','daily_avg'], axis=1)

# define regression function 
# provides several options for regression functions 
def regression(X_train, X_test, y_train, y_test):
    Regressor = {
        #'Random Forest Regressor': RandomForestRegressor(n_estimators=200),
        #'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=500),
        'ExtraTrees Regressor': ExtraTreesRegressor(n_estimators=500, min_samples_split=5),
        #'Bayesian Ridge': BayesianRidge(),
        #'Elastic Net CV': ElasticNetCV()
    }
    
    # fit models
    for name, clf in Regressor.items():
        print(name)
        clf.fit(X_train, y_train)
    
        # log loss data in console
        print(f'R2: {r2_score(y_test, clf.predict(X_test)):.2f}') #r^2
        print(f'MAE: {mean_absolute_error(y_test, clf.predict(X_test)):.2f}') # mean absolute error
        print(f'MSE: {mean_squared_error(y_test, clf.predict(X_test)):.2f}') # mean squared error

# run regression function with test/train data
regression(x_train_ETH, x_test_ETH, y_train_ETH, y_test_ETH)




# define prediction function
def prediction(name, X, y, X_forecast):
    model = ExtraTreesRegressor(n_estimators=500, min_samples_split=5)
    model.fit(X, y)
    target = model.predict(X_forecast)
    return target

# create array of predicted price data
forecasted_ETH = prediction('ETH', x_ETH, y_ETH, x_forecast_ETH)
# print(forecasted_ETH)

# converts floats to datetime object
data['time'] = pd.to_datetime(data['time'], unit='s')
print(data.info())

# define index for next 30 days
last_date=data.iloc[-1].time
print("last_date: " + str(last_date))
modified_date = last_date + dt.timedelta(days=1)
print("modified_date: " + str(modified_date))

# sets how many days to predict data for
new_date = pd.date_range(modified_date,periods=168,freq='D')
print("new_date: " + str(new_date))

# dataframe for predictions with new date as index
forecasted_ETH = pd.DataFrame(forecasted_ETH, columns=['daily_avg'], index=new_date)
data = data.set_index('time')
print(forecasted_ETH)

# combine historical data and predictions
ethereum = pd.concat([data[['daily_avg']], forecasted_ETH])
print(ethereum)

# creates matlpotlib figure from daily average data
plt.figure(figsize=(15,8))
(ethereum[22000:-168]['daily_avg']).plot(label='Historical Price')
(ethereum[-169:]['daily_avg']).plot(label='Predicted Price')

# labels axes, titles, creates legend, and shows final graph
plt.xlabel('Time')
plt.ylabel('Price in USD')
plt.title('Prediction on Daily Average Price of Ethereum')
plt.legend()
plt.show()