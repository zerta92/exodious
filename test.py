import bot_plots as bot
from firebase_test import send_metrics, send_error, send_notifications, update_firebase_snapshot, get_latest_metrics, get_latest_long_or_short, send_ig_info
from utils_test import send_ig_info_to_firebase, send_metrics_to_firebase, send_notifications_to_firebase, get_latest_metrics_from_firebase, get_latest_long_or_short_from_firebase

import numpy as np
# import matplotlib as plt
# import plotly as py
# import plotly.graph_objs as go
import btalib

import pandas as pd


# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

import tkinter as tk
from tkinter import filedialog


def read_csv_file():
    file_path = filedialog.askopenfilename(title=' Please choose File ')
    file = pd.read_csv(file_path)
    return file


def test_make_trades():
    strategy = bot.EMAStrategy()
    strategy.slow = 0.9
    strategy.fast = 0.8
    strategy.last_completed_candle = 0.85
    strategy.make_trades()

# test_make_trades()


def test_get_granular_data():
    strategy = bot.Strategy()
    return strategy.get_granular_data()


def calc_emas_test():
    strategy = bot.Strategy()
    return strategy.calc_emas()


def get_emas(df):
    data = df
    data_flipped = data[::-1]
    ema_fast = btalib.ema(data_flipped, period=5,
                          _seed=data_flipped['Close'].iloc[0]).df['ema']
    ema_slow = btalib.ema(data_flipped, period=12,
                          _seed=data_flipped['Close'].iloc[0]).df['ema']
    return [ema_fast, ema_slow]


def calc_rsi(price, RSI_SETTING, current):

    delta = price['Close']
    print('deltaa-----')
    print(delta)
    price_with_current = pd.concat(
        [delta, pd.DataFrame({0: [current]})])
    print(price_with_current)
    diff = delta.diff()
    dUp, dDown = diff.copy(), diff.copy()
    dUp[dUp < 0] = 0
    dDown[dDown > 0] = 0
    RolUp = dUp.rolling(RSI_SETTING).mean()  # pd.rolling_mean(dUp, n)
    # pd.rolling_mean(dDown, n).abs()
    RolDown = dDown.rolling(RSI_SETTING).mean().abs()
    RS = RolUp / RolDown
    RS = RS.fillna(value=0)
    rsi = 100.0 - (100.0 / (1.0 + RS))

    return rsi


# def calc_ema(close, fast, slow):
#     ema_fast = btalib.ema(close, period=fast,
#                           _seed=close['Close'].iloc[0]).df['ema']
#     ema_slow = btalib.ema(close, period=slow,
#                           _seed=close['Close'].iloc[0]).df['ema']
#     return [ema_fast, ema_slow]


def createRandomData(seed):
    np.random.seed(seed)

    # Generate random data
    data = {
        # 100 random samples from a standard normal distribution
        'Open': np.random.randn(100),
        # 100 random integers between 1 and 100
        'Close': np.random.randint(1, 100, size=100),
        # 100 random choices from the given list
        'High': np.random.randint(1, 100, size=100),
        # 100 random choices from the given list
        'Low': np.random.randint(1, 100, size=100)
    }

    # Create a DataFrame
    df = pd.DataFrame(data)
    return df


# def plot(data):
    data = data.reset_index()
    data = data.drop(columns='Time')
    data = data.drop(columns='Local time')
    data = data.drop(columns='RSI')

    print(data)
    fig = plt.pyplot.figure()

    # for frame in [ema_fast, ema_slow, rsi]:
    #     plt.pyplot.plot(frame)
    plt.pyplot.plot(data)

    plt.pyplot.show()


def calculate_ema(data, close_column='Close', ema_period=14, current_exchange=None):
    df = data.copy()  # Avoid modifying the original DataFrame
    if current_exchange:
        print('here')
        close_values = df[close_column].tolist()
        close_values.append(current_exchange)
        new_df = pd.DataFrame(close_values)
        ema = new_df.ewm(span=ema_period, adjust=False).mean()
        print(ema)
        return ema.iloc[-1]
    print('here2')
    ema = data[close_column].ewm(span=ema_period, adjust=False).mean()
    print(ema)
    return ema.iloc[-1]


random_data = createRandomData(42)
# print(random_data)
ema = calculate_ema(random_data, close_column='Close',
                    ema_period=14, current_exchange=90)

print(ema)


# csv = read_csv_file()
# csv = csv.set_index('Local time')
# print(csv)

# data = pd.DataFrame({'Time': csv.index, 'RSI': rsi, 'EMA_FAST': ema_fast,
#                     'EMA_SLOW': ema_slow, 'CLOSE': csv['Close']})
# print('FINAL')
# print(data)
# plot(data)
# # data.to_csv('timmy_3.csv')
