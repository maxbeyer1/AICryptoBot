import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('./data/ETHUSDT_1h.csv', parse_dates=['time'],index_col='time')
print(data.head())

plt.figure(figsize=(15,8))
(data['close']).plot(color='blue', label='close')

plt.legend()
plt.xlabel('time')
plt.show()