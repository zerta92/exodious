import logging


def log_emas(ema_fast, ema_slow, prev_ema_fast, prev_ema_slow):
    logging.critical('fast EMA ' + str(ema_fast))
    logging.critical('slow EMA ' + str(ema_slow))
    logging.critical('prev fast EMA ' + str(prev_ema_fast))
    logging.critical('prev slow EMA ' + str(prev_ema_slow))


def log_rsi(previous_rsi, current_rsi):
    logging.critical("current RSI: " + str(previous_rsi))
    logging.critical("prev RSI: " + str(current_rsi))


def log_make_trades_data(ema_fast, ema_slow, current_rsi, previous_rsi, in_long, in_short):
    logging.critical(' FAST: '+str(ema_fast))
    logging.critical(' SLOW: '+str(ema_slow))
    logging.critical(' RSI: '+str(current_rsi))
    logging.critical('PREV_RSI:' + str(previous_rsi))
    logging.critical("in_long: "+str(in_long))
    logging.critical("in_short: "+str(in_short))
