# https://code.visualstudio.com/docs/python/python-tutorial
# import numpy as np
import pandas as pd
import json
import btalib
# import mplfinance as mpf

# import matplotlib as m
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import threading
import time

import logging
# from datetime import datetime
import schedule
import requests

from utils_test import calculate_ema, snake_case_to_proper_case, send_notifications_to_firebase, send_error_to_firebase

API_KEY = '3B01QWGPBVQBKLG2'
BACKUP_API_KEY = 'R6ZSU4QDSJ052XQN'
FROM_CURRENCY = 'BTC'
TO_CURRENCY = 'USD'
SHORT_WINDOW = 'DIGITAL_CURRENCY_WEEKLY'
LONG_WINDOW = 'DIGITAL_CURRENCY_MONTHLY'

STREAM_INTERVAL = 2  # min
ANIMATION_STREAM = (STREAM_INTERVAL + 0.1)*60 * 1000  # millisec

FAST = 20
SLOW = 25
RSI_SETTING = 5
OVERBOUGHT = 66.66
OVERSOLD = 33.33
INSTRUMENT = 'USD_GBP'
BUY_UNITS = '3000'
SELL_UNITS = '3000'
BUY_TRANSACTION_FEE = .00428
SELL_TRANSACTION_FEE = .00348

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

        self.last_completed_candle = prev_data_df.iloc[-1]
        self.in_long = False
        self.in_short = False
        self.ema_begin_trading_flag = False
        self.force_buy = False
        self.force_sell = False

        # RSI
        self.current_rsi = 0
        rsi = self.calc_rsi(prev_data_df)
        print(rsi)
        self.current_rsi = rsi.iloc[-1]
        self.previous_rsi = rsi.iloc[-1]

        # EMA Short Window
        self.fast = 0
        self.slow = 0

        ema_fast_short = calculate_ema(
            prev_data_df, close_column='Close', ema_period=FAST)['EMA']
        ema_slow_short = calculate_ema(
            prev_data_df, close_column='Close', ema_period=SLOW)['EMA']
        self.prev_fast_short = ema_fast_short.iloc[-1]
        self.prev_slow_short = ema_slow_short.iloc[-1]

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
        last_close = float(self.last_completed_candle['Close'])
        self.y1.append(float(last_close))
        self.x1.append(time)
        self.line.set_data(self.x1, self.y1)

        self.axs[0].plot(self.x_buys, self.buy_marks, marker='D', color='lime')
        self.axs[0].plot(self.x_sells, self.sell_marks,
                         marker='X', color='red')

        self.y2.append(float(self.prev_slow_short))
        self.x2.append(time)
        self.line2.set_data(self.x2, self.y2)

        self.y3.append(float(self.prev_fast_short))
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
        rsi_signal = self.get_rsi_signal(
            self.current_rsi, self.previous_rsi)
        bid_price = exchange_rate
        ask_price = exchange_rate
        mid = (bid_price + ask_price)/2

        logging.critical(' FAST: '+str(self.fast))
        logging.critical(' SLOW: '+str(self.slow))
        logging.critical(' RSI: '+str(self.current_rsi))

        logging.critical('GBP:' + str(self.gbp))
        logging.critical('USD:' + str(self.usd))
        logging.critical("in_long: "+str(self.in_long))
        logging.critical("in_short: "+str(self.in_short))

        if self.fast > self.slow and rsi_signal == 'BUY':
            if self.in_short:
                self.in_short = False
                logging.critical('CLOSE ' + INSTRUMENT + ' ' + json.loads(BUY)
                                 ['order']['units'] + ' units at ask: ' + str(mid))
            if not self.in_long:
                buy_units = float(json.loads(BUY)['order']['units'])
                fee = buy_units*(BUY_TRANSACTION_FEE)
                self.in_long = True
                self.gbp = self.gbp + buy_units
                self.usd = self.usd - float(mid)*buy_units
                self.profit = self.profit - fee

                #  Graph points
                self.last_buy = mid
                self.buy_marks.append(self.last_buy)
                self.x_buys.append(self.index_counter)

                logging.critical('BUY ' + INSTRUMENT + ' ' + json.loads(BUY)
                                 ['order']['units'] + ' units at bid: ' + str(mid))
                send_notifications_to_firebase(
                    {'buy': True, 'sell': False, 'mid': mid, 'begin_trading': self.ema_begin_trading_flag})

        if (self.fast < self.slow and rsi_signal == 'SELL'):
            if self.in_long:
                self.in_long = False
                logging.critical('CLOSE ' + INSTRUMENT + ' ' + json.loads(SELL)
                                 ['order']['units'] + ' units at bid: ' + str(mid))
            if not self.in_short:
                sell_units = float(json.loads(SELL)['order']['units']) * -1
                fee = sell_units*(SELL_TRANSACTION_FEE)
                self.in_short = True
                self.gbp = self.gbp - sell_units
                self.usd = self.usd + float(mid)*float(sell_units)
                self.profit = self.profit + \
                    ((mid - self.last_buy) * sell_units) - fee
                self.last_sell = mid
                self.sell_marks.append(self.last_sell)
                self.x_sells.append(self.index_counter)
                self.force_sell = False

                logging.critical('SELL ' + INSTRUMENT + ' ' + json.loads(SELL)
                                 ['order']['units'] + ' units at bid: ' + str(mid))
                logging.critical('GBP:' + str(self.gbp))
                logging.critical('USD:' + str(self.usd))
                logging.critical('PROFIT:' + str(self.profit))
                logging.critical('FEE:' + str(fee))
                send_notifications_to_firebase(
                    {'buy': False, 'sell': True, 'mid': mid, 'begin_trading': self.ema_begin_trading_flag})

    def check_if_meet_trade_parameters(self, mid, transaction):
        print('SELL CHECK')
        print('LAST BUY PRICE:' + str(self.last_buy))
        print('SELL PRICE:' + str(mid))
        # check if at least 1% gain
        is_more_than_fee = False
        # if transaction == 'BUY':
        #     meets_one_percent = mid * 0.99 >= float(self.last_buy)
        #     print('BUY PRICE:' + str(ask_price))
        if transaction == 'SELL':
            is_more_than_fee = mid - \
                float(self.last_buy) > (SELL_TRANSACTION_FEE)

        return is_more_than_fee

    # Handles functions for every next candle closed
    def on_candle_close(self, prev_data_df_short, exchange_rate):
        logging.critical(
            'LAST CLOSE '+str(prev_data_df_short.iloc[-1]['Close']))

        last_close = prev_data_df_short.iloc[-1]['Close']

        # short window chart
        # EMA
        ema_short_calculations = self.calc_emas_short(
            prev_data_df_short, exchange_rate)

        ema_slow = ema_short_calculations[0]
        ema_fast = ema_short_calculations[1]

        # RSI
        self.previous_rsi = self.current_rsi
        rsi = self.calc_rsi(prev_data_df_short)
        rsi_delta = self.check_rsi_delta(self.current_rsi, rsi.iloc[-1])
        logging.critical("RSI: " + str(rsi.iloc[-1]))
        self.current_rsi = rsi.iloc[-1]
        print('RSI: ', self.current_rsi)

        if float(last_close) != exchange_rate:
            self.last_completed_candle = prev_data_df_short.iloc[-1]
            if (self.ema_begin_trading_flag):
                self.make_trades(exchange_rate)

    def calc_ema(self, period, prev_ema, last_close_price):
        '''Calculates the current EMA given a close price and the previous EMA'''
        return (last_close_price - prev_ema) * (2 / (period + 1)) + prev_ema

    def calc_emas_short(self, prev_data_df_short, exchange_rate):
        last_close = exchange_rate
        # Calculate EMAs of last closed candlestick
        self.fast = calculate_ema(
            prev_data_df_short, close_column='Close', ema_period=FAST, current_exchange=last_close)
        self.slow = calculate_ema(
            prev_data_df_short, close_column='Close', ema_period=SLOW, current_exchange=last_close)
        logging.critical('FAST EMA ' + str(self.fast))
        logging.critical('SLOW EMA ' + str(self.slow))
        # check if first cross has occurred to begin trading
        if (not self.ema_begin_trading_flag):
            if ((self.prev_fast_short > self.prev_slow_short and self.fast < self.slow) or (self.prev_fast_short < self.prev_slow_short and self.fast > self.slow)):
                self.ema_begin_trading_flag = True
        # set prev EMAs to current EMAs for next candlestick's calculation
        self.prev_fast_short = self.fast
        self.prev_slow_short = self.slow
        return [self.slow, self.fast]

    def get_rsi_signal(self, current_, previous_):
        current = float(current_)
        previous = float(previous_)
        delta = previous-current
        if current >= OVERSOLD and previous < OVERSOLD and current >= OVERSOLD:
            return 'BUY'
        if current <= OVERBOUGHT and previous > OVERBOUGHT and current != 0:
            return 'SELL'
        # Failure Swing: Bottom
        if current >= OVERBOUGHT and previous < OVERBOUGHT and current >= OVERBOUGHT and not self.in_long:
            return 'BUY'
        # Failure Swing: Top
        if current <= OVERSOLD and previous > OVERSOLD and current != 0:
            self.deffered_until_crossover = False
            if self.in_long:
                return 'SELL'
        if delta <= -8 and self.in_long:  # and current > OVERBOUGHT
            send_error_to_firebase("Delta dropped more than 8%")
            self.deffered_until_crossover = True
            return 'SELL'
        return ''

    def calc_rsi(self, price):
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

    def check_rsi_delta(self, current_, new_):
        new = float(new_)
        current = float(current_)
        delta = new-current
        if (delta >= 50 or delta <= -40):
            self.force_sell = True
        return delta

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
            self.fast, self.last_completed_candle), interval=400)
        plt.tight_layout()
        plt.show()


def get_data(strategy, key):
    try:
        prev_data_df_short = strategy.get_granular_data(SHORT_WINDOW, key)
        exchange_rate = strategy.get_current_data()
        strategy.on_candle_close(
            prev_data_df_short, exchange_rate)
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
    # ani = FuncAnimation(strategy.fig, strategy.animate,
    #                     interval=ANIMATION_STREAM)

    # Begin worker on a different thread
    thread = threading.Thread(target=scheduler, args=(strategy,), daemon=False)
    thread.start()
