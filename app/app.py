import pandas as pd

data = pd.read_csv('./data/ETHUSDT_1h.csv', parse_dates=['time'],index_col='time')
print(data.head())