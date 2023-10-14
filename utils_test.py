

import pandas as pd
from datetime import datetime
from firebase_test import send_metrics, send_error, send_notifications, update_firebase_snapshot, get_latest_metrics, get_latest_long_or_short, send_ig_info


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
        ema = new_df.ewm(span=ema_period, adjust=False).mean()
        return ema.iloc[-1][0]
    ema = data[close_column].ewm(span=ema_period, adjust=False).mean()
    return ema.iloc[-1]


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
