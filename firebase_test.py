import time
from datetime import timedelta, datetime
from uuid import uuid4

import firebase_admin
from firebase_admin import firestore, initialize_app, credentials


# Use a service account
# export GOOGLE_APPLICATION_CREDENTIALS='/home/django/django_project/exodious-749b2-firebase-adminsdk-l63jg-603dc760e6.json' -> to make env variable
cred = credentials.Certificate(
    '/Users/opteo/Documents/EXODIOUS/FOREX/exodious-749b2-firebase-adminsdk-l63jg-603dc760e6.json')
firebase_admin.initialize_app(cred)

__all__ = ['send_metrics','send_error', 'send_ig_info',
           'update_firebase_snapshot', 'send_notifications', "get_latest_metrics", "get_latest_long_or_short"]

# initialize_app()


def send_metrics(raw_metrics):
    print(raw_metrics)
    db = firestore.client()
    start = time.time()
    db.collection('metrics').document(str(uuid4())).create(raw_metrics)
    end = time.time()
    spend_time = timedelta(seconds=end - start)
    return spend_time


def send_notifications(notifications):
    print(notifications)
    db = firestore.client()
    start = time.time()
    db.collection('notifications').document(str(uuid4())).create(notifications)
    end = time.time()
    spend_time = timedelta(seconds=end - start)
    return spend_time

def send_error(error):
    date = datetime.now()
    error= str(error)
    error_obj = {'date':date,'error':error}
    db = firestore.client()
    db.collection('errors').document(str(uuid4())).create(error_obj)

def send_ig_info(info):
    date = datetime.now()
    db = firestore.client()
    info_obj = {'date':date,'info':info}
    db.collection('ig_account_info').document(str(uuid4())).create(info_obj)


def update_firebase_snapshot(snapshot_id):
    start = time.time()
    db = firestore.client()
    db.collection('notifications').document(snapshot_id).update(
        {'is_read': True}
    )
    end = time.time()
    spend_time = timedelta(seconds=end - start)
    return spend_time

def get_latest_metrics():
    ordered_metrics= []
    try :
        db = firestore.client()
        ordered_metrics = db.collection('metrics').order_by('date',direction=firestore.Query.DESCENDING).limit(1).get()
        # ordered_metrics = querySnapshot.orderBy('date', 'desc').limit(1).get()
    except:
        print('error getting latest metrics')

    if (ordered_metrics):
        metrics = list(map(lambda metric: metric.to_dict(), ordered_metrics))
        return metrics[0]
    
def get_latest_long_or_short():
    notifications = []
    try:
        db = firestore.client()
        notifications = db.collection('notifications').order_by(
            'date', direction=firestore.Query.DESCENDING).limit(1).get()

    except:
        print('error getting latest metrics')

    if (notifications):
        latest_notification = list(map(lambda notification: notification.to_dict(), notifications))
        return latest_notification[0]


    
def formatMetrics(metrics):
    ema_fast_15 = map(metrics, lambda metric: metric.ema_fast)
    ema_fast_60 = map(metrics,  lambda metric: metric.ema_fast_60)
    ema_slow_15 =map(metrics,  lambda metric:  metric.ema_slow_15)
    ema_slow_60 =map(metrics,  lambda metric:  metric.ema_slow_60)
    rsi_15 = map(metrics,  lambda metric:  metric.rsi_15)
    rsi_60 = map(metrics,  lambda metric:  metric.rsi_60)
    last_close_15 = map(metrics,  lambda metric:  metric.last_close_15)
    last_close_60 = map(metrics,  lambda metric:  metric.last_close_60)

    return {
        "last_close_15": last_close_15,
        "last_close_60": last_close_60,
        "ema_fast_15": ema_fast_15,
        "ema_fast_60": ema_fast_60,
        "ema_slow_15":ema_slow_15,
        "ema_slow_60":ema_slow_60,
        "rsi_15":rsi_15,
        "rsi_60": rsi_60,
    }


print(get_latest_long_or_short())