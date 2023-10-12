import numpy as np
import pandas as pd
import json
import btalib
import mplfinance as mpf

import matplotlib as m
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import threading
import time

import logging
from datetime import datetime
import schedule
import requests  

API_KEY = '3B01QWGPBVQBKLG2'
BACKUP_API_KEY = 'R6ZSU4QDSJ052XQN'
FROM_CURRENCY = 'GBP'
TO_CURRENCY = 'USD'
FIFTEEN_MINUTE_INTERVAL = '15min'
SIXTY_MINUTE_INTERVAL = '60min'

STREAM_INTERVAL = 2 #min
ANIMATION_STREAM = (STREAM_INTERVAL + 0.1)*60* 1000 #millisec

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

#put dots on graph when buy and sell
#https://www.investopedia.com/terms/e/ema.asp
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
       
     
        #GENERAL
        prev_data_df = self.get_granular_data(FIFTEEN_MINUTE_INTERVAL, API_KEY)   
        self.last_completed_candle = prev_data_df.iloc[-1]
        self.in_long = False
        self.in_short = False
        self.begin_trading = False
        self.force_buy = False
        self.force_sell= False
       
        #RSI
        self.current_rsi=0
        rsi = self.calc_rsi(prev_data_df)
        self.current_rsi = rsi.iloc[-1]
       
        #EMA 15
        self.fast = 0
        self.slow = 0
        ema_fast = btalib.ema(prev_data_df, period=FAST, _seed=prev_data_df['Close'].iloc[0]).df['ema']
        ema_slow = btalib.ema(prev_data_df, period=SLOW, _seed=prev_data_df['Close'].iloc[0]).df['ema']
        self.prev_fast = ema_fast.iloc[-1]
        self.prev_slow = ema_slow.iloc[-1]
        
        #EMA 60
        last_60_chart = self.get_granular_data(SIXTY_MINUTE_INTERVAL, API_KEY)
        self.fast_60 = 0
        self.slow_60 = 0
        ema_fast_60 = btalib.ema(last_60_chart, period=FAST, _seed=last_60_chart['Close'].iloc[0]).df['ema']
        ema_slow_60 = btalib.ema(last_60_chart, period=SLOW, _seed=last_60_chart['Close'].iloc[0]).df['ema']
        self.prev_fast_60 = ema_fast_60.iloc[-1]
        self.prev_slow_60 = ema_slow_60.iloc[-1]

        #LOG
        self.total = float(5000)
        self.profit =float(0)
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
        self.y1=[]
        self.line, = self.axs[0].plot([], [], label='Close')
       
        self.x2 = []
        self.y2=[]
        self.line2, = self.axs[0].plot([], [], label="Slow EMA" ,color='darkred')
       
        self.x3 = []
        self.y3=[]
        self.line3, = self.axs[0].plot([], [], label="Fast EMA" ,color='darkgreen')
        self.axs[0].legend(['close', 'slow ema', 'fast ema'], loc='upper left')
        self.lines=[[],[]]
       
        self.x4 = []
        self.y4=[]
        self.axs[1].set_xlabel('time')
        self.axs[1].set_ylabel('RSI')
        self.axs[1].axhline(y=OVERBOUGHT, color = 'gray')
        self.axs[1].axhline(y=OVERSOLD, color = 'gray')


        
        self.index_counter = 0
        self.x_buys=[]
        self.x_sells=[]
        
        self.buy_marks=[]
        self.sell_marks =[]
       
       
       
    def animate(self,i):
            time = i
            self.index_counter = time
            last_close = float(self.last_completed_candle['Close'])
            self.y1.append(float(last_close))
            self.x1.append(time)
            self.line.set_data(self.x1, self.y1)
            
            self.axs[0].plot(self.x_buys,self.buy_marks, marker='D', color='lime')
            self.axs[0].plot(self.x_sells,self.sell_marks, marker='X', color='red')
           
            self.y2.append(float(self.prev_slow))
            self.x2.append(time)
            self.line2.set_data(self.x2, self.y2)
           
            self.y3.append(float(self.prev_fast))
            self.x3.append(time)
            self.line3.set_data(self.x3, self.y3)
            
            self.axs[0].set_ylim(last_close*0.998,last_close*1.002)
            self.axs[0].set_xlim(0,i+1)
            
            self.y4.append(float(self.current_rsi))
            self.x4.append(time)
            
            self.axs[1].set_ylim(0,100)
            self.axs[1].set_xlim(0,i+1)

            color = 'tab:pink'
            self.axs[1].plot(self.x4, self.y4, color=color)
            self.axs[1].tick_params(axis='y', labelcolor=color)
                
               
    def make_trades(self):
        r = self.get_current_data()
        bid_price = float(r["8. Bid Price"][0])
        ask_price = float(r["9. Ask Price"][0])
        mid = (float(bid_price) + float(ask_price))/2
        buy_and_sell_stat = self.do_rsi_checks(self.current_rsi)
       
        logging.critical(' FAST: '+str(self.fast))
        logging.critical(' SLOW: '+str(self.slow))
        logging.critical(' RSI: '+str(self.current_rsi))
          
        logging.critical('GBP:' + str(self.gbp))
        logging.critical('USD:' + str(self.usd))
        logging.critical("in_long: "+str(self.in_long))
        logging.critical("in_short: "+str(self.in_short))
        

        if self.fast > self.slow and self.usd >= float(BUY_UNITS) and buy_and_sell_stat[0]:    
            if self.in_short:
                self.in_short = False             
                logging.critical('CLOSE ' + INSTRUMENT + ' ' + json.loads(BUY)['order']['units'] + ' units at ask: ' + str(mid) )
            if not self.in_long:
                buy_units = float(json.loads(BUY)['order']['units'])
                fee = buy_units*(BUY_TRANSACTION_FEE)
                self.in_long = True
                self.gbp = self.gbp + buy_units
                self.usd = self.usd  - float(mid)*buy_units
                self.profit = self.profit - fee
                self.last_buy = mid
                self.buy_marks.append(self.last_buy)
                self.x_buys.append(self.index_counter)

                logging.critical('BUY ' + INSTRUMENT + ' ' + json.loads(BUY)['order']['units'] + ' units at bid: ' + str(mid) )   
                logging.critical('GBP:' + str(self.gbp))
                logging.critical('USD:' + str(self.usd))
                logging.critical('FEE:' + str(fee))
                logging.critical('PROFIT:' + str(self.profit))
        if ( ((self.fast < self.slow and buy_and_sell_stat[1]) or (self.force_sell and self.in_long)) and self.gbp >= float(SELL_UNITS)) :
            if self.in_long:
                self.in_long = False
                logging.critical('CLOSE ' + INSTRUMENT + ' ' + json.loads(SELL)['order']['units'] + ' units at bid: ' + str(mid) )
            if not self.in_short:
                sell_units = float(json.loads(SELL)['order']['units'])* -1
                fee = sell_units*(SELL_TRANSACTION_FEE)
                self.in_short = True
                self.gbp = self.gbp - sell_units
                self.usd = self.usd + float(mid)*float(sell_units)
                self.profit = self.profit + (( mid - self.last_buy ) * sell_units) - fee
                self.last_sell = mid
                self.sell_marks.append(self.last_sell)
                self.x_sells.append(self.index_counter)
                self.force_sell=False
                
                logging.critical('SELL ' + INSTRUMENT + ' ' + json.loads(SELL)['order']['units'] + ' units at bid: ' + str(mid) )          
                logging.critical('GBP:' + str(self.gbp))
                logging.critical('USD:' + str(self.usd))
                logging.critical('PROFIT:' + str(self.profit))
                logging.critical('FEE:' + str(fee))




               
    def check_if_meet_trade_parameters(self, mid, transaction):
            print('SELL CHECK')
            print('LAST BUY PRICE:' + str(self.last_buy))
            print('SELL PRICE:' + str(mid))
            #check if at least 1% gain
            is_more_than_fee = False
            # if transaction == 'BUY':
            #     meets_one_percent = mid * 0.99 >= float(self.last_buy)
            #     print('BUY PRICE:' + str(ask_price))
            if transaction == 'SELL':
                is_more_than_fee = mid - float(self.last_buy) > (SELL_TRANSACTION_FEE) 

           
            return is_more_than_fee
               

   
    def get_interval(self):
        api_url_forex_intraday = 'https://www.alphavantage.co/query?function=FX_INTRADAY&from_symbol={}&to_symbol={}&interval={}&outputsize=full&apikey={}'.format(FROM_CURRENCY,TO_CURRENCY,FIFTEEN_MINUTE_INTERVAL,API_KEY)
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
        
         
        #15 min chart
        
        self.calc_emas_15(last_15_chart)
        rsi = self.calc_rsi(last_15_chart)
        rsi_delta = self.check_rsi_delta(self.current_rsi, rsi.iloc[-1])
        logging.critical("RSI: "+ str(rsi.iloc[-1]))
        self.current_rsi = rsi.iloc[-1]
        print('RSI: ', self.current_rsi)

        #60 min chart
        
        self.calc_emas_60(last_60_chart)
        rsi_60_data = self.calc_rsi(last_60_chart)
        rsi_60 = rsi_60_data.iloc[-1]
        buy_and_sell_stat = self.do_rsi_checks(rsi_60)
        sixty_min_buy_check=False
        
        if(self.fast_60 > self.slow_60):
            sixty_min_buy_check = True
        print('BEGIN trading: ', self.begin_trading)
        print('SIXTY MIN CHECK: ', sixty_min_buy_check)
        print('RIGHT SIDE: ', float(self.last_completed_candle["Close"]))
        print('LEFT SIDE: ', float(last_close))
        if float(last_close) != float(self.last_completed_candle["Close"]): 
            self.last_completed_candle = last_15_chart.iloc[-1]
            self.last_completed_candle_60 = last_60_chart.iloc[-1]
            if (self.begin_trading and sixty_min_buy_check):
                self.make_trades()

    def calc_ema(self, period, prev_ema, last_close_price):
        '''Calculates the current EMA given a close price and the previous EMA'''
        return (last_close_price - prev_ema) * (2 / (period + 1)) + prev_ema
   
    def calc_emas_15(self, last):
        last_completed_candle = last.iloc[-1]
        last_close = float(last_completed_candle["Close"])
        # Calculate EMAs of last closed candlestick
        self.fast = self.calc_ema(FAST, self.prev_fast, last_close)
        self.slow = self.calc_ema(SLOW, self.prev_slow, last_close)
        logging.critical('FAST EMA ' +  str(self.fast))
        logging.critical('SLOW EMA ' + str(self.slow))
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
        logging.critical('FAST EMA 60 ' +  str(self.fast_60))
        logging.critical('SLOW EMA 60 ' + str(self.slow_60))
        # set prev EMAs to current EMAs for next candlestick's calculation
        self.prev_fast_60 = self.fast_60
        self.prev_slow_60 = self.slow_60
        

    def do_rsi_checks(self, rsi):
        buy = rsi > OVERBOUGHT
        sell = rsi < OVERSOLD
        return [buy,sell]


         
    def calc_rsi(self, price):
        delta = price['Close'].diff()
        dUp, dDown = delta.copy(), delta.copy()
        dUp[dUp < 0] = 0
        dDown[dDown > 0] = 0
        RolUp = dUp.rolling(RSI_SETTING).mean()#pd.rolling_mean(dUp, n)
        RolDown = dDown.rolling(RSI_SETTING).mean().abs()#pd.rolling_mean(dDown, n).abs()
        RS = RolUp / RolDown
        rsi= 100.0 - (100.0 / (1.0 + RS))
        
        return rsi
    
    def check_rsi_delta(self, current_, new_):
        new= float(new_)
        current = float(current_)
        delta = new-current
        if (delta >= 50 or delta <= -40):
            self.force_sell=True
        return delta
        
               
        
             
    def get_granular_data(self, interval, key):
        api_url_forex_intraday = 'https://www.alphavantage.co/query?function=FX_INTRADAY&from_symbol={}&to_symbol={}&interval={}&outputsize=full&apikey={}'.format(FROM_CURRENCY,TO_CURRENCY,interval,key)
        exchange_rate_interval = requests.get(api_url_forex_intraday)
        if exchange_rate_interval.status_code == 200:
            raw_data_string = json.dumps(exchange_rate_interval.json())
            raw_data_json = json.loads(raw_data_string)
            df_raw_data = pd.DataFrame(raw_data_json["Time Series FX ({})".format(interval)])
            data = self.format_data((df_raw_data))
            return data[::-1]

        else:
             print('REQUEST_ERROR')
             
    def get_current_data(self):
        api_url_forex = 'https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={}&to_currency={}&apikey={}'.format(FROM_CURRENCY,TO_CURRENCY,API_KEY)
        response = requests.get(api_url_forex).json()
        raw_data_string = json.dumps(response)
        raw_data_json = json.loads(raw_data_string)
        df_json_raw = pd.DataFrame(raw_data_json["Realtime Currency Exchange Rate"], index=[0])
        return df_json_raw
       
       
    def get_daily_data(self):
        api_url_forex_intraday = 'https://www.alphavantage.co/query?function=FX_DAILY&from_symbol={}&to_symbol={}&outputsize=full&apikey={}'.format(FROM_CURRENCY,TO_CURRENCY,API_KEY)
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
        #Switch rows and columns
        df = df_raw_data.T
        latest = df.iloc[0]
        return latest
   
    def format_data(self, data):
        #Switch rows and columns
        df = data.T
        #Rename columns
        df["date"] = df.index
        df["Open"] = df["1. open"]
        df["Close"] = df["4. close"]
        df["High"] = df["2. high"]
        df["Low"] = df["3. low"]
        candlestick_ready_data = pd.DataFrame({'Open':df.Open,'Close':df.Close, 'High': df.High, 'Low': df.Low})
        candlestick_ready_data['Gmt time'] = pd.to_datetime(df['date'],format='%Y.%m.%d')
        df_reset_index = candlestick_ready_data.reset_index(drop=True)
        df_reset_index = df_reset_index.set_index(df_reset_index['Gmt time'])
        df = df_reset_index.drop_duplicates(keep=False)
        #Turn values to numbers
        df["Open"] = pd.to_numeric(df["Open"], downcast="float")
        df["Close"] = pd.to_numeric(df["Close"], downcast="float")
        df["High"] = pd.to_numeric(df["High"], downcast="float")
        df["Low"] = pd.to_numeric(df["Low"], downcast="float")
        return df
   
    def setup_plot(self):
        self.ani = FuncAnimation(self.fig, self.live_plot, fargs=(self.fast, self.last_completed_candle), interval=400)
        plt.tight_layout()
        plt.show()
       

   
def get_data(strategy, key):
    try:
        last_15_chart = strategy.get_granular_data(FIFTEEN_MINUTE_INTERVAL, key)
        last_60_chart = strategy.get_granular_data(SIXTY_MINUTE_INTERVAL, key)
        strategy.on_candle_close(last_15_chart, last_60_chart)
    except:
        get_data(strategy, BACKUP_API_KEY)



def scheduler(strategy):
    #TASK SCHEDULER
    schedule.every(STREAM_INTERVAL).minutes.do(get_data, strategy=strategy, key= API_KEY)

    while True:
        schedule.run_pending()
        time.sleep(1)
       

if __name__ == '__main__':
    strategy = Strategy()
    #PLOT
    ani = FuncAnimation(strategy.fig, strategy.animate, interval=ANIMATION_STREAM)
   
    x = threading.Thread(target=scheduler, args=(strategy,), daemon=True)
    x.start()
    