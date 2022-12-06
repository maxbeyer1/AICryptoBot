import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge, ElasticNetCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

data = pd.read_csv('./data/ETHUSDT_1h.csv', parse_dates=['time'], index_col='time')

print(data.index)

print(data.columns.tolist())

def datetime_to_float(d):
    epoch = dt.datetime.utcfromtimestamp(0)
    total_seconds =  (d - epoch).total_seconds()
    # total_seconds will be in decimals (millisecond precision)
    return total_seconds

data['time'] = pd.to_datetime(data['time'], unit='s')
# data['time'] = data['time'].astype(int)
data['time'] = data['time'].apply(lambda x: datetime_to_float(x))

print(data.head())
print(data.info())

# close price graphs
plt.figure(figsize=(15,8))
(data['close']).plot(color='blue', label='close')

plt.legend()
plt.xlabel('time')
plt.show()

eth = data['close']

# 200 day moving average
eth_ma = eth.rolling(window=200).mean()

# get close prices
close = pd.concat([eth], axis=1)
close_ma = pd.concat([eth_ma], axis=1)
close_ma.tail()

# plot moving average for closing price for cryptocurrencies
close_ma.plot(figsize=(15,8))
plt.title('5-Day Moving Average on Daily Closing Price')
plt.xlabel('time')
plt.ylabel('price in USD')
plt.show()

# calculate daily average price
data['daily_avg'] = (data['open'] + data['high'] + data['low'] + data['close']) / 4
data['daily_avg_After_Month']=data['daily_avg'].shift(-30)
x_ETH = data.dropna().drop(['daily_avg_After_Month','daily_avg'], axis=1)
y_ETH = data.dropna()['daily_avg_After_Month']
x_train_ETH, x_test_ETH, y_train_ETH, y_test_ETH = train_test_split(x_ETH, y_ETH, test_size=0.2, random_state=43)
x_forecast_ETH =  data.tail(30).drop(['daily_avg_After_Month','daily_avg'], axis=1)

# define regression function
def regression(X_train, X_test, y_train, y_test):
    Regressor = {
        #'Random Forest Regressor': RandomForestRegressor(n_estimators=200),
        #'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=500),
        'ExtraTrees Regressor': ExtraTreesRegressor(n_estimators=500, min_samples_split=5),
        #'Bayesian Ridge': BayesianRidge(),
        #'Elastic Net CV': ElasticNetCV()
    }

    for name, clf in Regressor.items():
        print(name)
        clf.fit(X_train, y_train)
    
        print(f'R2: {r2_score(y_test, clf.predict(X_test)):.2f}')
        print(f'MAE: {mean_absolute_error(y_test, clf.predict(X_test)):.2f}')
        print(f'MSE: {mean_squared_error(y_test, clf.predict(X_test)):.2f}')
        print()


regression(x_train_ETH, x_test_ETH, y_train_ETH, y_test_ETH)

# define prediction function
def prediction(name, X, y, X_forecast):
    model = ExtraTreesRegressor(n_estimators=500, min_samples_split=5)
    model.fit(X, y)
    target = model.predict(X_forecast)
    return target

forecasted_ETH = prediction('ETH', x_ETH, y_ETH, x_forecast_ETH)

print(forecasted_ETH)

data['time'] = pd.to_datetime(data['time'], unit='s')
print(data.info())

# define index for next 30 days
last_date=data.iloc[-1].time
print("last_date: " + str(last_date))
modified_date = last_date + dt.timedelta(days=1)
print("modified_date: " + str(modified_date))

new_date = pd.date_range(modified_date,periods=30,freq='D')

print("new_date: " + str(new_date))

forecasted_ETH = pd.DataFrame(forecasted_ETH, columns=['daily_avg'], index=new_date)

ethereum = pd.concat([data[['daily_avg']], forecasted_ETH])

plt.figure(figsize=(15,8))
(ethereum[:-30]['daily_avg']).plot(label='Historical Price')
(ethereum[-31:]['daily_avg']).plot(label='Predicted Price')

plt.xlabel('Time')
plt.ylabel('Price in USD')
plt.title('Prediction on Daily Average Price of Ethereum')
plt.legend()
plt.show()