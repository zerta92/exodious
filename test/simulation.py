

# run from main path with python -m test.simulation

from utils.utils import get_emas, get_rsi_signal, get_ema_signal, get_ema_signal_crossover, calculate_ema, calc_rsi, snake_case_to_proper_case
import sys
from .test_data import data
import pandas as pd
sys.path.append("..")


FAST = 5
SLOW = 15
RSI_SETTING = 4
OVERBOUGHT = 66.66
OVERSOLD = 33.33
SHORT_WINDOW = 'DIGITAL_CURRENCY_DAILY'


def get_data_points():
    df = pd.DataFrame(
        data["Time Series ({})".format(snake_case_to_proper_case(SHORT_WINDOW))])
    return format_data(df)


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

    return df


def analyze_trades(trades):
    successful_trades = 0
    unsuccessful_trades = 0
    current_buy = None

    for trade in trades:
        if 'buy' in trade:
            current_buy = trade['buy']
        elif 'sell' in trade and current_buy is not None:
            if trade['sell'] > current_buy:
                successful_trades += 1
                current_buy = None
            else:
                unsuccessful_trades += 1
    return {'successful_trades': successful_trades, 'unsuccessful_trades': unsuccessful_trades}


def log_metrics(exchange_rate,
                last_close,
                ema_fast,
                ema_slow,
                prev_ema_fast,
                prev_ema_slow,
                previous_rsi,
                current_rsi,
                ema_signal_crossover, log=False):
    if not log:
        return

    print('current: ', exchange_rate)
    print('last close: ', last_close)
    print('ems fast: ', ema_fast)
    print('ema slow: ', ema_slow)
    print('prev ema fast: ', prev_ema_fast)
    print('prev ema slow: ', prev_ema_slow)
    print('prev rsi: ', previous_rsi)
    print('current rsi: ', current_rsi)
    print('ema crossover signal: ', ema_signal_crossover)


def run(df):
    starting_amount_usd = 100_000
    initial_values_start = 16
    values = df['Close']

    first_values = pd.DataFrame({
        'Close': values.head(initial_values_start),
    })

    last_close = float(first_values.iloc[-1]['Close'])

    # RSI
    current_rsi = 0
    rsi = calc_rsi(first_values, RSI_SETTING)
    current_rsi = rsi.iloc[-1]
    previous_rsi = rsi.iloc[-1]

    # EMA
    ema_fast_short = calculate_ema(
        first_values, close_column='Close', ema_period=FAST)
    ema_slow_short = calculate_ema(
        first_values, close_column='Close', ema_period=SLOW)

    ema_fast = ema_fast_short
    ema_slow = ema_slow_short
    prev_ema_fast = ema_fast_short
    prev_ema_slow = ema_slow_short

    in_long = False
    in_short = False

    trades = []

    stock_amount = 0
    current_amount_usd = starting_amount_usd
    for index, exchange_rate in enumerate(values.tail(len(values)-initial_values_start).tolist()):

        last_close = float(df.iloc[initial_values_start+index-1]['Close'])

        sliced_values_df = pd.DataFrame({
            'Close': values.head(16+index),
        })
        # RSI
        rsi = calc_rsi(sliced_values_df, RSI_SETTING)
        previous_rsi = current_rsi
        current_rsi = rsi.iloc[-1]

        # EMA
        ema_with_current_rate = get_emas(sliced_values_df, exchange_rate)
        ema_without_current_rate = get_emas(sliced_values_df)
        ema_slow = ema_with_current_rate[0]
        ema_fast = ema_with_current_rate[1]
        prev_ema_slow = ema_without_current_rate[0]
        prev_ema_fast = ema_without_current_rate[1]

        ema_signal_crossover = get_ema_signal_crossover(
            ema_fast, ema_slow, prev_ema_fast, prev_ema_slow)

        log_metrics(exchange_rate, last_close, ema_fast, ema_slow, prev_ema_fast,
                    prev_ema_slow, previous_rsi, current_rsi, ema_signal_crossover, log=False)

        if last_close != exchange_rate and ema_signal_crossover:
            rsi_signal = get_rsi_signal(current_rsi, previous_rsi)
            ema_signal = get_ema_signal(ema_fast, ema_slow)

            # Combination of RSI and EMA signals
            if ema_signal == 'BUY' and rsi_signal == 'BUY':
                if not in_long:
                    in_short = False
                    in_long = True
                    amount_to_buy = int(current_amount_usd/exchange_rate)
                    cost = amount_to_buy * exchange_rate
                    current_amount_usd -= cost
                    stock_amount += amount_to_buy
                    trades.append(
                        {'buy': exchange_rate, 'usd': current_amount_usd, 'amount_to_buy': amount_to_buy})

            elif ema_signal == 'SELL' and rsi_signal == 'SELL':
                if not in_short:
                    in_short = True
                    in_long = False
                    amount_to_sell = stock_amount
                    revenue = amount_to_sell * exchange_rate
                    current_amount_usd += revenue
                    stock_amount -= amount_to_sell
                    trades.append(
                        {'sell': exchange_rate, 'usd': current_amount_usd, 'amount_to_sell': amount_to_sell})

            elif ema_signal == 'SELL' and rsi_signal == 'BUY':
                if in_long:
                    in_short = True
                    in_long = False
                    amount_to_sell = stock_amount
                    revenue = amount_to_sell * exchange_rate
                    current_amount_usd += revenue
                    stock_amount -= amount_to_sell
                    trades.append(
                        {'sell': exchange_rate, 'usd': current_amount_usd, 'amount_to_sell': amount_to_sell})

            elif ema_signal == 'BUY' and rsi_signal == 'SELL':
                if in_short:
                    in_short = False
                    in_long = True
                    amount_to_buy = int(current_amount_usd/exchange_rate)
                    cost = amount_to_buy * exchange_rate
                    current_amount_usd -= cost
                    stock_amount += amount_to_buy
                    trades.append(
                        {'buy': exchange_rate, 'usd': current_amount_usd, 'amount_to_buy': amount_to_buy})

    successfull_trades = analyze_trades(trades)
    print('----------------------------')
    print(successfull_trades)
    print('starting_amount_usd: ', starting_amount_usd)
    print('current_amount_usd: ', current_amount_usd)
    print('profit: ', current_amount_usd-starting_amount_usd)


if __name__ == '__main__':
    data = get_data_points()
    run(data)
