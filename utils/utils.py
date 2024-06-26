

from .firebase_utils import send_metrics, send_error, send_notifications, update_firebase_snapshot, get_latest_metrics, get_latest_long_or_short, send_ig_info
import pandas as pd
from datetime import datetime

from variables import OVERBOUGHT, OVERSOLD, FAST, SLOW

__all__ = ['send_metrics_to_firebase', 'send_error_to_firebase',
           'send_notifications_to_firebase', "get_latest_metrics_from_firebase", "get_latest_long_or_short_from_firebase"]


def send_metrics_to_firebase(raw_metrics):
    ema_fast = raw_metrics['ema_fast']
    ema_slow = raw_metrics['ema_slow']
    rsi_15 = raw_metrics['rsi']
    ema_fast_60 = raw_metrics['fast_60']
    ema_slow_60 = raw_metrics['slow_60']
    rsi_60 = raw_metrics['rsi_60']
    last_close_15 = raw_metrics['last_close_15']
    date = datetime.now()
    metrics = {'last_close_15': last_close_15, 'rsi_15': rsi_15, 'rsi_60': rsi_60, 'ema_fast': ema_fast, 'ema_slow': ema_slow,
               'ema_fast_60': ema_fast_60, 'ema_slow_60': ema_slow_60, 'date': date}
    send_metrics(metrics)


def send_notifications_to_firebase(notifications):
    buy = notifications['buy']
    sell = notifications['sell']
    mid = notifications['mid']
    date = datetime.now()
    notifications = {'buy': buy, 'sell': sell, 'mid': mid, 'date': date}
    send_notifications(notifications)


def send_error_to_firebase(error):
    send_error(error)


def get_latest_metrics_from_firebase():
    return get_latest_metrics()


def get_latest_long_or_short_from_firebase():
    return get_latest_long_or_short()


def send_ig_info_to_firebase(info):
    return send_ig_info(info)


def calculate_ema(data, close_column='Close', ema_period=14, current_exchange=None):
    df = data.copy()  # Avoid modifying the original DataFrame
    if current_exchange:
        close_values = df[close_column].tolist()
        close_values.append(current_exchange)
        new_df = pd.DataFrame(close_values)
        ema = new_df.ewm(span=ema_period, adjust=True).mean()
        return ema.iloc[-1][0]
    ema = data[close_column].ewm(span=ema_period, adjust=True).mean()
    return ema.iloc[-1]


def get_emas(prev_ema_df, exchange_rate=None, fast_period=FAST, slow_period=SLOW):
    ema_fast = calculate_ema(
        prev_ema_df, close_column='Close', ema_period=fast_period, current_exchange=exchange_rate)
    ema_slow = calculate_ema(
        prev_ema_df, close_column='Close', ema_period=slow_period, current_exchange=exchange_rate)
    return [ema_slow, ema_fast]


def get_ema_signal_crossover(fast, slow, prev_fast, prev_slow):
    # check if first cross has occurred to begin trading
    if ((prev_fast > prev_slow and fast < slow) or (prev_fast < prev_slow and fast > slow)):
        return True
    return False


def get_ema_signal(fast, slow):
    # Determine EMA signal based on the relationship between fast and slow EMAs
    if fast > slow:
        return 'BUY'
    elif fast < slow:
        return 'SELL'
    else:
        return 'HOLD'


def get_rsi_signal(current_rsi, previous_rsi):
    current = float(current_rsi)
    previous = float(previous_rsi)
    delta = previous-current
    if current >= OVERSOLD and previous < OVERSOLD and current >= OVERSOLD:
        return 'BUY'
    if current <= OVERBOUGHT and previous > OVERBOUGHT and current != 0:
        return 'SELL'
    # Failure Swing: Bottom
    if current >= OVERBOUGHT and previous < OVERBOUGHT and current >= OVERBOUGHT:
        return 'BUY'
    # Failure Swing: Top
    if current <= OVERSOLD and previous > OVERSOLD and current != 0:
        return 'SELL'
    if delta <= -8:  # and current > OVERBOUGHT
        return 'SELL'
    return ''


def calc_rsi(price, RSI_SETTING):
    delta = price['Close']
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


def snake_case_to_proper_case(input_string):
    words = input_string.lower().split('_')
    proper_case_words = [word.capitalize() for word in words]
    result_string = ' '.join(proper_case_words)
    return result_string


def check_keys_for_string(json_obj, target_string):
    for key in json_obj.keys():
        if target_string in key:
            return True
    return False


def format_data(data):
    # Switch rows and columns
    df = data.T
    # Rename columns
    df["date"] = df.index
    df["Open"] = df["1b. open (USD)"]
    df["Close"] = df["4b. close (USD)"]
    df["High"] = df["2b. high (USD)"]
    df["Low"] = df["3b. low (USD)"]

    candlestick_ready_data = pd.DataFrame(
        {'Open': df.Open, 'Close': df.Close, 'High': df.High, 'Low': df.Low})
    candlestick_ready_data['Gmt time'] = pd.to_datetime(
        df['date'], format='%Y-%m-%d')
    df_reset_index = candlestick_ready_data.reset_index(drop=True)
    df_reset_index = df_reset_index.set_index(df_reset_index['Gmt time'])
    df = df_reset_index.drop_duplicates(keep=False)
    # Turn values to numbers
    df["Open"] = pd.to_numeric(df["Open"], downcast="float")
    df["Close"] = pd.to_numeric(df["Close"], downcast="float")
    df["High"] = pd.to_numeric(df["High"], downcast="float")
    df["Low"] = pd.to_numeric(df["Low"], downcast="float")

    return df.iloc[::-1]
