

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
    """
    Calculate Exponential Moving Average (EMA) for a given column in a DataFrame.

    Parameters:
        - data: DataFrame containing the financial data.
        - close_column: Name of the column for which EMA needs to be calculated.
        - ema_period: Time period for EMA calculation (default is 14).

    Returns:
        DataFrame with an additional column 'EMA' containing EMA values.
    """
    df = data.copy()  # Avoid modifying the original DataFrame

    # Calculate EMA using a simple Python loop
    alpha = 2 / (ema_period + 1)
    # Initial value is the first close price
    ema_values = [df[close_column].iloc[0]]

    for i in range(1, len(df)):
        if current_exchange is not None:
            # Adjust EMA calculation based on current exchange
            ema = alpha * df[close_column].iloc[i] + \
                (1 - alpha) * ema_values[-1] * current_exchange
        else:
            ema = alpha * df[close_column].iloc[i] + \
                (1 - alpha) * ema_values[-1]
        ema_values.append(ema)

    df['EMA'] = ema_values

    return df


def snake_case_to_proper_case(input_string):
    words = input_string.lower().split('_')
    proper_case_words = [word.capitalize() for word in words]
    result_string = ' '.join(proper_case_words)
    return result_string
