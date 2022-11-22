import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('./data/ETHUSDT_1h.csv', parse_dates=['time'],index_col='time')
print(data.head())

# close price graphs
plt.figure(figsize=(15,8))
(data['close']).plot(color='blue', label='close')

plt.legend()
plt.xlabel('time')
# plt.show()

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