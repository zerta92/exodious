# import numpy as np
import pandas as pd
import json
import btalib
# import mplfinance as mpf

# import matplotlib as m
# import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import threading
import time

import logging
from datetime import timedelta, datetime
import schedule
import requests
from utils_test import send_ig_info_to_firebase, send_metrics_to_firebase, send_notifications_to_firebase, get_latest_metrics_from_firebase, get_latest_long_or_short_from_firebase
from ig import buy_order, sell_order, login, confirm_order, get_positions, close_position, get_market_details


API_KEY = 'R6ZSU4QDSJ052XQN'
BACKUP_API_KEY = '3B01QWGPBVQBKLG2'
FROM_CURRENCY = 'GBP'
TO_CURRENCY = 'USD'
FIFTEEN_MINUTE_INTERVAL = '15min'
SIXTY_MINUTE_INTERVAL = '60min'

STREAM_INTERVAL = 10  # min
ANIMATION_STREAM = (STREAM_INTERVAL + 0.1)*60 * 1000  # millisec

FAST = 20  # 5
SLOW = 50  # 15
RSI_SETTING = 12  # was 5
OVERBOUGHT = 66
OVERSOLD = 30
RSI_DECREASE_THRESHHOLD = 0.85
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


# https://www.investopedia.com/terms/e/ema.asp
# https://www.investopedia.com/terms/r/rsi.asp


class Strategy():
    def __init__(self):
        '''Init function to initialize logger, variables for trading, and previous EMA for continuing data'''
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(format='%(asctime)s: %(message)s', datefmt='%m/%d/%Y %H:%M:%S %p', level=logging.CRITICAL, handlers=[
            logging.FileHandler("LOG_BOT.log"),
            logging.StreamHandler()
        ])
        logging.critical('Init')

        # Firebase
        self.data_loss = True

        # IG
        self.buy_error = False
        self.sell_error = False
        self.last_profit = 0

        # GENERAL
        prev_data_df = self.get_granular_data(SIXTY_MINUTE_INTERVAL, API_KEY)
        last_daily_chart = self.get_daily_data(API_KEY)
        self.last_completed_candle = prev_data_df.iloc[-1]
        self.previous_close = 0
        self.in_long = False
        self.begin_trading = False
        self.begin_trading_overwrite = False
        self.force_buy = False
        self.force_sell = False
        self.deffered_until_crossover = False
        self.up_count = 0
        self.down_count = 0

        # RSI
        rsi = self.calc_rsi(prev_data_df, RSI_SETTING)
        self.previous_rsi = rsi
        self.current_rsi = rsi

        self.current_rsi_60 = rsi

        self.last_15_timestamp_check = datetime.now()
        ema_fast = btalib.ema(prev_data_df, period=FAST,
                              _seed=prev_data_df['Close'].iloc[0]).df['ema']
        ema_slow = btalib.ema(prev_data_df, period=SLOW,
                              _seed=prev_data_df['Close'].iloc[0]).df['ema']
        self.fast = ema_fast.iloc[-1]
        self.slow = ema_slow.iloc[-1]
        self.prev_fast = ema_fast.iloc[-1]
        self.prev_slow = ema_slow.iloc[-1]

        # EMA 60
        self.last_60_timestamp_check = last_daily_chart.iloc[-1]['Gmt time']
        ema_fast_60 = btalib.ema(
            last_daily_chart, period=FAST, _seed=last_daily_chart['Close'].iloc[0]).df['ema']
        ema_slow_60 = btalib.ema(
            last_daily_chart, period=SLOW, _seed=last_daily_chart['Close'].iloc[0]).df['ema']
        self.fast_60 = ema_fast_60.iloc[-1]
        self.slow_60 = ema_slow_60.iloc[-1]
        self.prev_fast_60 = ema_fast_60.iloc[-1]
        self.prev_slow_60 = ema_slow_60.iloc[-1]

        # LOG
        self.total = float(5000)
        self.profit = float(0)
        self.spend = float(0)
        self.usd = float(5000)
        self.gbp = float(0)
        self.last_buy = float(0)
        self.last_sell = float(0)

    def make_trades(self):
        rsi_signal = self.get_rsi_signal(
            self.current_rsi, self.current_rsi_60, self.previous_rsi)
        last_close = float(self.last_completed_candle['Close'])
        logging.critical(' ON MAKE TRADES----- ')
        logging.critical('  PREVIOUS CLOSE: '+str(self.previous_close))
        logging.critical(' LAST CLOSE: '+str(last_close))
        logging.critical(' FAST: '+str(self.fast))
        logging.critical(' SLOW: '+str(self.slow))
        logging.critical(' CURRENT RSI: '+str(self.current_rsi))
        logging.critical(' CURRENT RSI dynamic: '+str(self.current_rsi_60))
        logging.critical(' RSI PREVIOUS: '+str(self.previous_rsi))
        logging.critical(' RSI SIGNAL: '+str(rsi_signal))
        logging.critical(' DOWN COUNT: '+str(self.down_count))
        logging.critical("in_long: "+str(self.in_long))
        logging.critical("deffered_until_crossover: " +
                         str(self.deffered_until_crossover))

        if self.force_buy or (not self.deffered_until_crossover and last_close > self.previous_close and not self.sell_error and (self.fast > self.slow and self.fast_60 > self.slow_60) and rsi_signal == 'BUY'):
            if self.force_buy or not self.in_long:
                buy_units = float(json.loads(BUY)['order']['units'])
                fee = buy_units*(BUY_TRANSACTION_FEE)
                self.in_long = True
                self.force_buy = False

                try:
                    self.buy_error = False
                    err = ''
                    login_info = login()
                    CST = login_info['CST']
                    X_SECURITY_TOKEN = login_info['X-SECURITY-TOKEN']
                    if not CST or not X_SECURITY_TOKEN:
                        err = 'BUY_ERROR UNABLE TO LOGIN'
                        logging.critical(err)
                        send_error_to_firebase(err)
                        self.buy_error = True
                        return
                    content = login_info['content']
                    account_info = content['accountInfo']
                    send_ig_info_to_firebase(account_info)

                    opened_long_positions = get_positions(
                        CST, X_SECURITY_TOKEN)['buy']
                    if len(opened_long_positions) > 0:
                        err = 'A BUY POSITION IS ALREADY OPEN'
                        logging.critical(err)
                        send_error_to_firebase(err)
                        return

                    deal_reference = buy_order(CST, X_SECURITY_TOKEN)
                    if not deal_reference:
                        self.buy_error = True
                        err = 'BUY_ERROR UNABLE TO BUY'
                        logging.critical(err)
                        send_error_to_firebase(err)
                        return

                    confirmation = confirm_order(
                        CST, X_SECURITY_TOKEN, deal_reference)
                    if not confirmation:
                        self.buy_error = True
                        err = 'BUY_ERROR NO CONFIRMATION'
                        logging.critical(err)
                        send_error_to_firebase(err)
                        return

                    self.buy_error = False

                except Exception as e:
                    logging.critical('BUY_ERROR')
                    self.buy_error = True
                    send_error_to_firebase(e)

                logging.critical('bought at :' + str(last_close))
                logging.critical('in long set to :' + str(self.in_long))

                send_notifications_to_firebase(
                    {'buy': True, 'sell': False, 'mid': last_close, 'begin_trading': True})
        if self.force_sell or (not self.buy_error and self.fast < self.slow or rsi_signal == 'SELL'):
            self.deffered_until_crossover = False
            if self.force_sell or self.in_long:
                self.in_long = False
                sell_units = float(json.loads(SELL)['order']['units']) * -1
                fee = sell_units*(SELL_TRANSACTION_FEE)
                self.force_sell = False

                try:
                    self.sell_error = False
                    err = ''
                    login_info = login()
                    CST = login_info['CST']
                    X_SECURITY_TOKEN = login_info['X-SECURITY-TOKEN']
                    if not CST or not X_SECURITY_TOKEN:
                        err = 'SELL_ERROR UNABLE TO LOGIN'
                        logging.critical(err)
                        send_error_to_firebase(err)
                        self.sell_error = True
                        return
                    content = login_info['content']
                    account_info = content['accountInfo']
                    send_ig_info_to_firebase(account_info)

                    deal_id = 0
                    buy_deal = get_positions(CST, X_SECURITY_TOKEN)['buy']
                    if len(buy_deal) != 0:
                        deal_id = buy_deal[0]['position']['dealId']
                    if deal_id != 0:
                        is_position_closed = close_position(
                            CST, X_SECURITY_TOKEN, deal_id, 'SELL')
                        if not is_position_closed:
                            err = 'UNABLE TO CLOSE POSITION'
                            self.sell_error = True
                            logging.critical(err)
                            send_error_to_firebase(err)
                    else:
                        err = 'SELL_ERROR NO POSITION TO CLOSE'
                        logging.critical(err)
                        send_error_to_firebase(err)
                        return

                except Exception as e:
                    logging.critical('SELL ERROR')
                    self.sell_error = True
                    send_error_to_firebase(e)

                logging.critical('sold at :' + str(last_close))
                logging.critical('in long set to :' + str(self.in_long))

                send_notifications_to_firebase(
                    {'buy': False, 'sell': True, 'mid': last_close, 'begin_trading': True})

    def get_interval(self):
        api_url_forex_intraday = 'https://www.alphavantage.co/query?function=FX_INTRADAY&from_symbol={}&to_symbol={}&interval={}&outputsize=full&apikey={}'.format(
            FROM_CURRENCY, TO_CURRENCY, SIXTY_MINUTE_INTERVAL, API_KEY)
        exchange_rate_interval = requests.get(api_url_forex_intraday)
        if exchange_rate_interval.status_code == 200:
            print('INTERVAL DATA')
        else:
            print('REQUEST_ERROR')

    def on_candle_close(self, last_15_chart, last_60_chart):
        '''Handles functions for every next candle closed'''
        # check if the current candle has ended, if so, assign it to last_completed_candle
        logging.critical('LAST CLOSE '+str(last_15_chart.iloc[-1]['Close']))
        last_close = last_15_chart.iloc[-1]['Close']
        date = datetime.now()
        weekday = datetime.today().weekday()
        hour = datetime.now().hour

        if (self.data_loss):
            logging.critical('DATA LOSS: ' + str(self.data_loss))
            send_error_to_firebase('Data Loss')
            # # last_saved_metrics = get_latest_metrics_from_firebase()
            latest_notification = get_latest_long_or_short_from_firebase()
            is_buy = latest_notification["buy"]
            is_begin_trading = latest_notification["begin_trading"]
            if is_begin_trading:
                self.begin_trading_overwrite = True
            if is_buy == True:
                self.in_long = True
                self.begin_trading_overwrite = True
            self.data_loss = False

        is_market_open = self.check_market_open(hour, weekday)
        if not is_market_open:
            return

        try:
            login_info = login()
            CST = login_info['CST']
            X_SECURITY_TOKEN = login_info['X-SECURITY-TOKEN']
            content = login_info['content']
            account_info = content['accountInfo']
            profit = float(account_info['profitLoss'])
            send_ig_info_to_firebase(account_info)
            positions = get_positions(CST, X_SECURITY_TOKEN)
            opened_long_positions = []
            if positions['positions']:
                opened_long_positions = positions['buy']
                if len(opened_long_positions) == 0 and self.in_long:
                    self.in_long = False
                    send_error_to_firebase('IN LONG WITH NO OPEN POSITIONS')
                    send_notifications_to_firebase(
                        {'buy': False, 'sell': True, 'mid': 0, 'begin_trading': True})
                    self.deffered_until_crossover = True
                if len(opened_long_positions) > 0 and not self.in_long:
                    self.in_long = True
                    send_error_to_firebase('IN SHORT WITH OPEN POSITION')
                    send_notifications_to_firebase(
                        {'buy': True, 'sell': False, 'mid': 0, 'begin_trading': True})
            else:
                send_error_to_firebase('FAILURE GETTING POSITIONS')

        except Exception as e:
            send_error_to_firebase(e)
            logging.critical('error getting profile info')

        rsi = self.calc_rsi(last_15_chart, RSI_SETTING)
        # self.get_latest_rsi(RSI_SETTING, '60min', API_KEY)
        self.current_rsi_60 = rsi
        # 15 min chart
        if (date >= self.last_15_timestamp_check + timedelta(days=1)):
            self.last_15_timestamp_check = date
            self.calc_emas_15(last_15_chart)
            self.previous_rsi = self.current_rsi
            self.current_rsi = rsi

        # 60 min chart
        if (self.last_60_timestamp_check != last_60_chart.iloc[-1]['Gmt time']):
            self.last_60_timestamp_check = last_60_chart.iloc[-1]['Gmt time']
            self.calc_emas_60(last_60_chart)

        raw_metrics = {'last_close_15': str(last_15_chart.iloc[-1]['Close']), 'ema_fast': self.fast, 'ema_slow': self.slow, 'rsi': self.current_rsi,
                       'fast_60': self.fast_60, 'slow_60': self.slow_60, 'rsi_60': rsi, 'date': date}
        send_metrics_to_firebase(raw_metrics)

        if float(last_close) != float(self.last_completed_candle["Close"]):
            self.previous_close = float(self.last_completed_candle["Close"])
            if float(last_close) < self.previous_close:
                self.down_count = self.down_count + 1
                self.up_count = 0
            else:
                self.down_count = 0
                self.up_count = self.up_count + 1
            self.last_completed_candle = last_15_chart.iloc[-1]
            self.last_completed_candle_60 = last_60_chart.iloc[-1]
            if ((self.begin_trading or self.begin_trading_overwrite)):
                self.make_trades()

    def check_market_open(self, hour, weekday):
        if hour > 23 or weekday == 5:
            return False
        if weekday == 4 and hour > 21:
            return False
        if weekday == 6 and hour < 21:
            return False
        return True

    def calc_ema(self, period, prev_ema, last_close_price):
        return (last_close_price - prev_ema) * (2 / (period + 1)) + prev_ema

    def calc_emas_15(self, last):
        last_completed_candle = last.iloc[-1]
        last_close = float(last_completed_candle["Close"])
        # Calculate EMAs of last closed candlestick
        self.fast = self.calc_ema(FAST, self.prev_fast, last_close)
        self.slow = self.calc_ema(SLOW, self.prev_slow, last_close)

        # check if first cross has occurred to begin trading
        if (not self.begin_trading):
            if ((self.prev_fast > self.prev_slow and self.fast < self.slow) or (self.prev_fast < self.prev_slow and self.fast > self.slow)):
                self.begin_trading = True
        # set prev EMAs to current EMAs for next candlestick's calculation
        self.prev_fast = self.fast
        self.prev_slow = self.slow

    def calc_emas_60(self, last):
        last_completed_candle = last.iloc[-1]
        last_close = float(last_completed_candle["Close"])
        # Calculate EMAs of last closed candlestick
        self.fast_60 = self.calc_ema(FAST, self.prev_fast_60, last_close)
        self.slow_60 = self.calc_ema(SLOW, self.prev_slow_60, last_close)

        # set prev EMAs to current EMAs for next candlestick's calculation
        self.prev_fast_60 = self.fast_60
        self.prev_slow_60 = self.slow_60

    def do_rsi_checks(self, rsi):
        buy = rsi > OVERBOUGHT
        sell = rsi < OVERSOLD
        return [buy, sell]

    def calc_rsi_old(self, price):
        delta = price['Close'].diff()
        dUp, dDown = delta.copy(), delta.copy()
        dUp[dUp < 0] = 0
        dDown[dDown > 0] = 0
        RolUp = dUp.rolling(RSI_SETTING).mean()  # pd.rolling_mean(dUp, n)
        # pd.rolling_mean(dDown, n).abs()
        RolDown = dDown.rolling(RSI_SETTING).mean().abs()
        RS = RolUp / RolDown
        rsi = 100.0 - (100.0 / (1.0 + RS))
        return rsi

    def calc_rsi(self, data, num):
        last_key = ''
        # data = data.tail(100)
        data = data.to_dict()['Close']
        if not isinstance(data, dict):
            raise Exception('Dictionary input expected')
        if not isinstance(num, int):
            raise Exception('Integer input expected')
        if num < 1 or num > 21:
            raise Exception('Unusual numeric input detected')
        if (num > len(data)):
            raise Exception('Insufficient data for calculation')
        data_keys = list(data.keys())
        data_list = list(data.values())
        result = {}
        last_price = -1
        gains_losses_list = []
        for x in range(len(data_list)):
            if (last_price != -1):
                diff = round((data_list[x] - last_price), 4)
                if (diff > 0):
                    gains_losses = [data_list[x], diff, 0]
                elif (diff < 0):
                    gains_losses = [data_list[x], 0, abs(diff)]
                else:
                    gains_losses = [data_list[x], 0, 0]

                gains_losses_list.append(gains_losses)
            sum_gains = 0
            sum_losses = 0
            avg_gains = 0
            avg_losses = 0
            rs = 0
            if (x == num):
                series = gains_losses_list[-num::]
                for y in series:
                    sum_gains += y[1]
                    sum_losses += y[2]
                try:
                    avg_gains = sum_gains / num
                    avg_losses = sum_losses / num
                    rs = avg_gains / avg_losses
                except:
                    rs = 0

                rsi = 100 - (100 / (1 + rs))
                last_gain_avg = avg_gains
                last_loss_avg = avg_losses
                result[data_keys[x]] = round(rsi, 2)
            if (x > num):
                current_list = gains_losses_list[-1::]
                current_gain = current_list[0][1]
                current_loss = current_list[0][2]
                try:
                    current_gains_avg = (
                        last_gain_avg * (num - 1) + current_gain) / num
                    current_losses_avg = (
                        last_loss_avg * (num - 1) + current_loss) / num
                    rs = current_gains_avg / current_losses_avg
                except:
                    rs = 0

                rsi = 100 - (100 / (1 + rs))
                last_gain_avg = current_gains_avg
                last_loss_avg = current_losses_avg
                if (x == len(data_list)-1):
                    last_key = data_keys[x]
                result[data_keys[x]] = round(rsi, 2)

            last_price = data_list[x]

        return result[last_key]

    def check_temp_ema(self, current_fast, current_slow, last_close):
        last_close = float(last_close)
        current_fast = float(current_fast)
        current_slow = float(current_slow)
        fast_ema = self.calc_ema(FAST, current_fast, last_close)
        slow_ema = self.calc_ema(SLOW, current_slow, last_close)

        if (fast_ema < slow_ema):
            return False
        return True

    def get_rsi_signal(self, current_, current_dynamic_, previous_):
        current = float(current_)
        current_dynamic = float(current_dynamic_)
        previous = float(previous_)
        delta = current_dynamic-current
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
        if self.down_count >= 3 and current_dynamic <= OVERSOLD and self.in_long:
            send_error_to_firebase(
                "Dropped into oversold after decreasing 3 periods")
            self.deffered_until_crossover = True
            return 'SELL'
        return ''

    # def get_rsi_signal(self, current_, current_dynamic_, previous_):
    #     current = float(current_)
    #     current_dynamic = float(current_dynamic_)
    #     previous = float(previous_)
    #     if current >= OVERSOLD and previous < OVERSOLD and current != 0:
    #         return 'BUY'
    #     if ((current <= OVERBOUGHT and previous > OVERBOUGHT and current != 0) or (current_dynamic <= OVERBOUGHT and previous > OVERBOUGHT)):
    #         return 'SELL'
    #     # # Failure Swing: Bottom
    #     # if ((current >= OVERBOUGHT and previous < OVERBOUGHT and current != 0) or (current_dynamic >= OVERBOUGHT and previous < OVERBOUGHT)) and not self.in_long:
    #     #     # if dynamic triggered, then move current rsi
    #     #     if not (current >= OVERBOUGHT and previous < OVERBOUGHT and current != 0) and (current_dynamic >= OVERBOUGHT and previous < OVERBOUGHT):
    #     #         self.current_rsi = current_dynamic
    #     #     return 'BUY'
    #     if current >= OVERBOUGHT and previous < OVERBOUGHT and current != 0 and not self.in_long:
    #         return 'BUY'
    #     # Failure Swing: Top
    #     if ((current <= OVERSOLD and previous > OVERSOLD and current != 0) or (current_dynamic <= OVERSOLD * RSI_DECREASE_THRESHHOLD and previous > OVERSOLD)) and self.in_long:
    #         # if dynamic triggered, then move current rsi
    #         # if not (current <= OVERSOLD and previous > OVERSOLD and current != 0) and (current_dynamic <= OVERSOLD and previous > OVERSOLD):
    #         #     self.current_rsi = current_dynamic
    #         return 'SELL'
    #     return ''

    def get_granular_data(self, interval, key):
        try:
            api_url_forex_intraday = 'https://www.alphavantage.co/query?function=FX_INTRADAY&from_symbol={}&to_symbol={}&interval={}&outputsize=full&apikey={}'.format(
                FROM_CURRENCY, TO_CURRENCY, interval, key)
            exchange_rate_interval = requests.get(api_url_forex_intraday)

            if exchange_rate_interval.status_code == 200:
                raw_data_string = json.dumps(exchange_rate_interval.json())
                raw_data_json = json.loads(raw_data_string)
                df_raw_data = pd.DataFrame(
                    raw_data_json["Time Series FX ({})".format(interval)])
                data = self.format_data((df_raw_data))
                return data[::-1]

            else:
                logging.critical('GET GRANULAR DATA REQUEST ERROR')
        except Exception as e:
            send_error_to_firebase(e)
            logging.critical('GET GRANULAR DATA REQUEST ERROR')

    def get_latest_rsi(self, rsi_setting, interval, key):
        try:
            rsi_raw_data_url = 'https://www.alphavantage.co/query?function=RSI&symbol=GBPUSD&interval={}&time_period={}&series_type=open&apikey={}'.format(
                interval, rsi_setting, key)
            rsi_raw_data = requests.get(rsi_raw_data_url)
            if rsi_raw_data.status_code == 200:
                raw_data_string = rsi_raw_data.json()
                df_raw_data = pd.DataFrame(
                    raw_data_string["Technical Analysis: RSI"])
                df = df_raw_data.T
                return float(df['RSI'].iloc[0])
            else:
                logging.critical('GET LATEST RSI REQUEST ERROR')
                return 0
        except Exception as e:
            send_error_to_firebase(e)
            logging.critical('GET LATEST RSI REQUEST ERROR')
            return 0

    def get_current_data(self):
        api_url_forex = 'https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={}&to_currency={}&apikey={}'.format(
            FROM_CURRENCY, TO_CURRENCY, API_KEY)
        response = requests.get(api_url_forex).json()
        raw_data_string = json.dumps(response)
        raw_data_json = json.loads(raw_data_string)
        df_json_raw = pd.DataFrame(
            raw_data_json["Realtime Currency Exchange Rate"], index=[0])
        return df_json_raw

    def get_daily_data(self, key):
        api_url_forex_intraday = 'https://www.alphavantage.co/query?function=FX_DAILY&from_symbol={}&to_symbol={}&outputsize=full&apikey={}'.format(
            FROM_CURRENCY, TO_CURRENCY, key)
        exchange_rate_interval = requests.get(api_url_forex_intraday)
        if exchange_rate_interval.status_code == 200:
            raw_data_string = json.dumps(exchange_rate_interval.json())
            raw_data_json = json.loads(raw_data_string)
            df_raw_data = pd.DataFrame(
                raw_data_json["Time Series FX ({})".format('Daily')])
            data = self.format_data((df_raw_data))
            return data[::-1]

    def format_data(self, data):
        # Switch rows and columns
        df = data.T
        # Rename columns
        df["date"] = df.index
        df["Open"] = df["1. open"]
        df["Close"] = df["4. close"]
        df["High"] = df["2. high"]
        df["Low"] = df["3. low"]
        candlestick_ready_data = pd.DataFrame(
            {'Open': df.Open, 'Close': df.Close, 'High': df.High, 'Low': df.Low})
        candlestick_ready_data['Gmt time'] = pd.to_datetime(
            df['date'], format='%Y.%m.%d')
        df_reset_index = candlestick_ready_data.reset_index(drop=True)
        df_reset_index = df_reset_index.set_index(df_reset_index['Gmt time'])
        df = df_reset_index.drop_duplicates(keep=False)
        # Turn values to numbers
        df["Open"] = pd.to_numeric(df["Open"], downcast="float")
        df["Close"] = pd.to_numeric(df["Close"], downcast="float")
        df["High"] = pd.to_numeric(df["High"], downcast="float")
        df["Low"] = pd.to_numeric(df["Low"], downcast="float")
        return df


def get_data(strategy, key):
    try:
        last_60_chart = strategy.get_granular_data(
            SIXTY_MINUTE_INTERVAL, key)
        # last_60_chart = strategy.get_granular_data(SIXTY_MINUTE_INTERVAL, key)
        last_daily_chart = strategy.get_daily_data(key)
        print(last_60_chart)
        print(last_daily_chart)
        return
        strategy.on_candle_close(last_60_chart, last_daily_chart)
    except Exception as e:
        logging.critical('sheduler error, retrying in 10 seconds...')
        send_error_to_firebase(e)
        time.sleep(20)
        get_data(strategy, BACKUP_API_KEY)


def scheduler(strategy):
    # TASK SCHEDULER
    send_error_to_firebase('STARTED')
    schedule.every(STREAM_INTERVAL).minutes.do(
        get_data, strategy=strategy, key=API_KEY)

    while True:
        schedule.run_pending()
        time.sleep(1)


def run():
    print('STARTING1...')
    strategy = Strategy()
    scheduler(strategy)


def run_once():
    print('STARTING2...')
    strategy = Strategy()
    get_data(strategy, API_KEY)


run_once()

# if __name__ == 'exodious.bot.bot':
#     print('STARTING...')
#     strategy = Strategy()
#     scheduler(strategy)
