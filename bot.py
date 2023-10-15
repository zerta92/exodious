import pandas as pd
import json

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import threading
import time

import logging

import schedule
import requests

from variables import FAST, SLOW, RSI_SETTING, OVERBOUGHT, OVERSOLD
from utils.logging_utils import log_make_trades_data, log_emas, log_rsi
from utils.utils import check_keys_for_string, get_ema_signal, get_rsi_signal, get_ema_signal_crossover, calculate_ema, get_emas, calc_rsi, snake_case_to_proper_case, send_notifications_to_firebase
from utils.firebase_utils import get_latest_long_or_short

API_KEY = 'R6ZSU4QDSJ052XQN'
BACKUP_API_KEY = '3B01QWGPBVQBKLG2'
FROM_CURRENCY = 'BTC'
TO_CURRENCY = 'USD'
SHORT_WINDOW = 'DIGITAL_CURRENCY_DAILY'
LONG_WINDOW = 'DIGITAL_CURRENCY_WEEKLY'

STREAM_INTERVAL = 30  # min
ANIMATION_STREAM = (STREAM_INTERVAL + 0.1) * 60 * 1000  # millisec

INSTRUMENT = 'USD_GBP'
BUY_UNITS = '3000'
SELL_UNITS = '3000'

# IG
BUY = json.dumps({
    "order": {
        "instrument": INSTRUMENT,
        "units": BUY_UNITS,
        "type": "MARKET",
        "positionFill": "DEFAULT"
    }
})

SELL = json.dumps({
    "order": {
        "instrument": INSTRUMENT,
        "units": '-' + SELL_UNITS,
        "type": "MARKET",
        "positionFill": "DEFAULT"
    }
})

# https://www.theforexchronicles.com/the-ema-5-and-ema-20-crossover-trading-strategy/

# put dots on graph when buy and sell
# https://www.investopedia.com/terms/e/ema.asp
# https://www.investopedia.com/terms/r/rsi.asp


class Strategy:
    def __init__(self):
        '''Init function to initialize logger, variables for trading, and previous EMA for continuing data'''
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(format='%(asctime)s: %(message)s', datefmt='%m/%d/%Y %H:%M:%S %p', level=logging.CRITICAL, handlers=[
            logging.FileHandler("LOG.log"),
            logging.StreamHandler()
        ])

        # GENERAL
        prev_data_df = self.get_granular_data(SHORT_WINDOW, API_KEY)

        self.last_completed_candle_df = prev_data_df.iloc[-1]
        self.in_long = get_latest_long_or_short()['buy']
        self.in_short = not self.in_long
        self.force_buy = False
        self.force_sell = False

        # RSI
        self.current_rsi = 0
        rsi = calc_rsi(prev_data_df, RSI_SETTING)
        self.current_rsi = rsi.iloc[-1]
        self.previous_rsi = rsi.iloc[-1]

        # EMA
        ema_fast_short = calculate_ema(
            prev_data_df, close_column='Close', ema_period=FAST)
        ema_slow_short = calculate_ema(
            prev_data_df, close_column='Close', ema_period=SLOW)
        self.ema_fast = ema_fast_short
        self.ema_slow = ema_slow_short
        self.prev_ema_fast = ema_fast_short
        self.prev_ema_slow = ema_slow_short

        # LOG
        self.total = float(5000)
        self.profit = float(0)
        self.spend = float(0)
        self.usd = float(5000)
        self.gbp = float(0)
        self.last_buy = float(0)
        self.last_sell = float(0)

        # #PLOT
        plt.style.use('fivethirtyeight')
        self.fig, self.axs = plt.subplots(2, 1)
        self.axs[0].set_xlabel('time')
        self.axs[0].set_ylabel('GBP')

        self.x1 = []
        self.y1 = []
        self.line, = self.axs[0].plot([], [], label='Close')

        self.x2 = []
        self.y2 = []
        self.line2, = self.axs[0].plot(
            [], [], label="Slow EMA", color='darkred')

        self.x3 = []
        self.y3 = []
        self.line3, = self.axs[0].plot(
            [], [], label="Fast EMA", color='darkgreen')
        self.axs[0].legend(['close', 'slow ema', 'fast ema'], loc='upper left')
        self.lines = [[], []]

        self.x4 = []
        self.y4 = []
        self.axs[1].set_xlabel('time')
        self.axs[1].set_ylabel('RSI')
        self.axs[1].axhline(y=OVERBOUGHT, color='gray')
        self.axs[1].axhline(y=OVERSOLD, color='gray')

        self.index_counter = 0
        self.x_buys = []
        self.x_sells = []

        self.buy_marks = []
        self.sell_marks = []

    def animate(self, i):
        time = i
        self.index_counter = time
        last_close = float(self.last_completed_candle_df['Close'])
        self.y1.append(float(last_close))
        self.x1.append(time)
        self.line.set_data(self.x1, self.y1)

        self.axs[0].plot(self.x_buys, self.buy_marks, marker='D', color='lime')
        self.axs[0].plot(self.x_sells, self.sell_marks,
                         marker='X', color='red')

        self.y2.append(float(self.prev_ema_slow))
        self.x2.append(time)
        self.line2.set_data(self.x2, self.y2)

        self.y3.append(float(self.prev_ema_fast))
        self.x3.append(time)
        self.line3.set_data(self.x3, self.y3)

        self.axs[0].set_ylim(last_close*0.998, last_close*1.002)
        self.axs[0].set_xlim(0, i+1)

        self.y4.append(float(self.current_rsi))
        self.x4.append(time)

        self.axs[1].set_ylim(0, 100)
        self.axs[1].set_xlim(0, i+1)

        color = 'tab:pink'
        self.axs[1].plot(self.x4, self.y4, color=color)
        self.axs[1].tick_params(axis='y', labelcolor=color)

    def make_trades(self, exchange_rate):
        rsi_signal = get_rsi_signal(self.current_rsi, self.previous_rsi)
        ema_signal = get_ema_signal(self.ema_fast, self.ema_slow,)
        log_make_trades_data(self.ema_fast, self.ema_slow, self.current_rsi,
                             self.previous_rsi, self.in_long, self.in_short)

        # Combination of RSI and EMA signals
        if ema_signal == 'BUY' and rsi_signal == 'BUY':
            if not self.in_long:
                self.execute_buy(exchange_rate)

        elif ema_signal == 'SELL' and rsi_signal == 'SELL':
            if not self.in_short:
                self.execute_sell(exchange_rate)

        elif ema_signal == 'SELL' and rsi_signal == 'BUY':
            if self.in_long:
                self.execute_sell(exchange_rate)

        elif ema_signal == 'BUY' and rsi_signal == 'SELL':
            if self.in_short:
                self.execute_buy(exchange_rate)

    def execute_sell(self, exchange_rate):
        self.in_short = True
        self.in_long = False
        self.sell_marks.append(exchange_rate)
        self.x_sells.append(self.index_counter)
        self.last_sell = exchange_rate
        logging.critical('SELL ' + INSTRUMENT + ' ' + json.loads(SELL)
                         ['order']['units'] + ' units at bid: ' + str(exchange_rate))
        send_notifications_to_firebase(
            {'buy': False, 'sell': True, 'mid': exchange_rate, 'begin_trading': True})

    def execute_buy(self, exchange_rate):
        self.in_short = False
        self.in_long = True
        self.buy_marks.append(exchange_rate)
        self.x_buys.append(self.index_counter)
        self.last_buy = exchange_rate

        logging.critical('BUY ' + INSTRUMENT + ' ' + json.loads(BUY)
                         ['order']['units'] + ' units at bid: ' + str(exchange_rate))
        send_notifications_to_firebase(
            {'buy': True, 'sell': False, 'mid': exchange_rate, 'begin_trading': True})

    def on_candle_close(self, prev_ema_df, exchange_rate):
        logging.critical(
            'Current exchange rate '+str(exchange_rate))

        last_close = float(prev_ema_df.iloc[-1]['Close'])
        self.last_completed_candle_df = prev_ema_df.iloc[-1]

        # EMA
        ema_with_current_rate = get_emas(
            prev_ema_df, exchange_rate)
        ema_without_current_rate = get_emas(prev_ema_df)

        self.ema_slow = ema_with_current_rate[0]
        self.ema_fast = ema_with_current_rate[1]
        self.prev_ema_slow = ema_without_current_rate[0]
        self.prev_ema_fast = ema_without_current_rate[1]
        log_emas(self.ema_fast, self.ema_slow,
                 self.prev_ema_fast, self.prev_ema_slow)

        ema_signal_crossover = get_ema_signal_crossover(
            self.ema_fast, self.ema_slow, self.prev_ema_fast, self.prev_ema_slow)

        # RSI
        rsi = calc_rsi(prev_ema_df, RSI_SETTING)
        self.previous_rsi = self.current_rsi
        self.current_rsi = rsi.iloc[-1]
        log_rsi(self.previous_rsi, self.current_rsi)

        if last_close != exchange_rate and ema_signal_crossover:
            self.make_trades(exchange_rate)

    def get_granular_data(self, window, key):
        api_url_forex_intraday = 'https://www.alphavantage.co/query?function={}&symbol={}&market={}&outputsize=full&apikey={}'.format(window,
                                                                                                                                      FROM_CURRENCY, TO_CURRENCY, key)
        exchange_rate_interval = requests.get(api_url_forex_intraday)
        if exchange_rate_interval.status_code == 200:
            raw_data_string = json.dumps(exchange_rate_interval.json())
            raw_data_json = json.loads(raw_data_string)
            df_raw_data = pd.DataFrame(
                raw_data_json["Time Series ({})".format(snake_case_to_proper_case(window))])
            data = self.format_data((df_raw_data))
            # Reverse dataframe
            return data[::-1]

        else:
            print('REQUEST_ERROR')

    def get_current_data(self):
        api_url_forex = 'https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={}&to_currency={}&apikey={}'.format(
            FROM_CURRENCY, TO_CURRENCY, API_KEY)
        response = requests.get(api_url_forex).json()
        raw_data_string = json.dumps(response)
        raw_data_json = json.loads(raw_data_string)
        df_json_raw = pd.DataFrame(
            raw_data_json["Realtime Currency Exchange Rate"], index=[0])
        return float(df_json_raw["5. Exchange Rate"][0])

    def get_daily_data(self):
        api_url_forex_intraday = 'https://www.alphavantage.co/query?function=FX_DAILY&from_symbol={}&to_symbol={}&outputsize=full&apikey={}'.format(
            FROM_CURRENCY, TO_CURRENCY, API_KEY)
        exchange_rate_interval = requests.get(api_url_forex_intraday)
        if exchange_rate_interval.status_code == 200:
            raw_data = exchange_rate_interval.json()
        else:
            print('REQUEST_ERROR')
        raw_data_string = json.dumps(raw_data)
        raw_data_json = json.loads(raw_data_string)
        try:
            df_raw_data = pd.DataFrame(raw_data_json["Time Series FX (Daily)"])
        except:
            print(raw_data_json)
        # Switch rows and columns
        df = df_raw_data.T
        latest = df.iloc[0]
        return latest

    def format_data(self, data):
        # Switch rows and columns
        df = data.T
        # Rename columns
        df["date"] = df.index
        df["Open"] = df["1a. open (USD)"]
        df["Close"] = df["4a. close (USD)"]
        df["High"] = df["2a. high (USD)"]
        df["Low"] = df["3a. low (USD)"]

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

    def setup_plot(self):
        self.ani = FuncAnimation(self.fig, self.live_plot, fargs=(
            self.ema_fast, self.last_completed_candle_df), interval=400)
        plt.tight_layout()
        plt.show()


def get_data(strategy, key):
    try:
        prev_ema_df = strategy.get_granular_data(SHORT_WINDOW, key)
        exchange_rate = strategy.get_current_data()
        strategy.on_candle_close(
            prev_ema_df, exchange_rate)
    except:
        # If error it could be due to api key usage, try a different key
        get_data(strategy, BACKUP_API_KEY)


def scheduler(strategy):
    schedule.every(STREAM_INTERVAL).minutes.do(
        get_data, strategy=strategy, key=API_KEY)

    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == '__main__':
    strategy = Strategy()
    # # PLOT
    ani = FuncAnimation(strategy.fig, strategy.animate,
                        interval=ANIMATION_STREAM)

    # Begin worker on a different thread
    thread = threading.Thread(target=scheduler, args=(strategy,), daemon=False)
    thread.start()
