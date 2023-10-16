
# run from main path with python -m test.simulation
import matplotlib.pyplot as plt
from utils.utils import get_emas, get_rsi_signal, get_ema_signal, get_ema_signal_crossover, calculate_ema, calc_rsi, snake_case_to_proper_case, check_keys_for_string
from .test_data import data_dot_daily as data
import pandas as pd


FAST = 6
SLOW = 61
RSI_SETTING = 16

SHORT_WINDOW = 'DIGITAL_CURRENCY_DAILY'
LONG_WINDOW = 'DIGITAL_CURRENCY_WEEKLY'


def get_data_points(data):

    is_short_window = check_keys_for_string(
        data, snake_case_to_proper_case(SHORT_WINDOW))

    window = SHORT_WINDOW if is_short_window else LONG_WINDOW

    df = pd.DataFrame(
        data["Time Series ({})".format(snake_case_to_proper_case(window))])
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

    return df.iloc[::-1]


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
    print('<---------------->')


def run(df, FAST=FAST, SLOW=SLOW, RSI_SETTING=RSI_SETTING):

    starting_amount_usd = 100000
    initial_values_start = 16
    values = df['Close']

    first_values = pd.DataFrame({
        'Close': values.head(initial_values_start),
    })

    first_values_prev = pd.DataFrame({
        'Close': values.head(initial_values_start-1),
    })

    last_close = float(first_values.iloc[-1]['Close'])

    # RSI
    current_rsi = 0
    rsi = calc_rsi(first_values, RSI_SETTING)
    current_rsi = rsi.iloc[-1]
    previous_rsi = rsi.iloc[-1]

    # EMA
    ema_fast = calculate_ema(
        first_values, close_column='Close', ema_period=FAST)
    ema_slow = calculate_ema(
        first_values, close_column='Close', ema_period=SLOW)

    prev_ema_fast = calculate_ema(
        first_values_prev, close_column='Close', ema_period=FAST)

    prev_ema_slow = calculate_ema(
        first_values_prev, close_column='Close', ema_period=SLOW)

    ema_signal_crossover = False

    in_long = False
    in_short = True
    trades = []
    stock_amount = 0
    current_amount_usd = starting_amount_usd

    # Graph Data
    exchange_rates = []
    buy_points = []
    sell_points = []
    ema_fast_values = []
    ema_slow_values = []

    # Iterate over data, skipping the initial values
    for index, exchange_rate in enumerate(values.tail(len(values)-initial_values_start).tolist()):

        sliced_values_df = pd.DataFrame({
            'Close': values.head(initial_values_start+index),
        })

        last_close = float(sliced_values_df['Close'].iloc[-1])

        # RSI
        rsi = calc_rsi(sliced_values_df, RSI_SETTING)
        previous_rsi = current_rsi
        current_rsi = rsi.iloc[-1]

        # EMA
        ema_with_current_rate = get_emas(
            sliced_values_df, exchange_rate, FAST, SLOW)

        prev_ema_slow = ema_slow
        prev_ema_fast = ema_fast
        ema_slow = ema_with_current_rate[0]
        ema_fast = ema_with_current_rate[1]

        ema_signal_crossover = True if ema_signal_crossover else get_ema_signal_crossover(
            ema_fast, ema_slow, prev_ema_fast, prev_ema_slow)

        log_metrics(exchange_rate, last_close, ema_fast, ema_slow, prev_ema_fast,
                    prev_ema_slow, previous_rsi, current_rsi, ema_signal_crossover, log=False)

        exchange_rates.append(exchange_rate)
        ema_fast_values.append(ema_fast)
        ema_slow_values.append(ema_slow)

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
                    buy_points.append(index)
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
                    sell_points.append(index)
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
                    sell_points.append(index)
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
                    buy_points.append(index)
                    trades.append(
                        {'buy': exchange_rate, 'usd': current_amount_usd, 'amount_to_buy': amount_to_buy})

    profit = current_amount_usd-starting_amount_usd
    # successfull_trades = analyze_trades(trades)
    # print('starting_amount_usd: ', starting_amount_usd)
    # print('current_amount_usd: ', current_amount_usd)
    print('profit: ', profit)

    # Plot Data
    plt.plot(exchange_rates)
    plt.scatter(buy_points, [exchange_rates[i]
                for i in buy_points], color='blue', marker='o', label='BUY')
    plt.scatter(sell_points, [exchange_rates[i]
                for i in sell_points], color='red', marker='o', label='SELL')
    plt.plot(ema_fast_values, label='EMA Fast', linestyle='--', color='green')
    plt.plot(ema_slow_values, label='EMA Slow', linestyle='--', color='orange')
    plt.xlabel('Time')
    plt.ylabel('Exchange Rate')
    plt.title('Bitcoin Exchange Rate Over Time')
    plt.show()
    return profit


def run_simulation():
    formatted_data = get_data_points(data)
    run(formatted_data)


if __name__ == '__main__':
    run_simulation()
