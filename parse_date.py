from datetime import datetime
from datetime import time as dtime
import re
import numpy as np

# 21:09:06-05:00
regex1 = r"^([0-9]|0[0-9]|1[0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]-05:00$"
# PM 7:59:16
regex2 = r"^(AM|PM)([0-9]|1[0-9]):([0-5][0-9]|[0-9]):([0-5][0-9]|[0-9])$"
# 9:01:04 PM / 9:03:25 AM
regex3 = r"^([0-9]|[1-2][0-2]):([0-5][0-9]|[0-9]):([0-5][0-9]|[0-9])(AM|PM)$"
# 23:53:46
regex4 = r"^([0-9]|0[0-9]|1[0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]$"


def try_parsing_date(text):
    for fmt in ('%d-%m-%Y', '%d.%m.%Y', '%m/%d/%Y', '%d-%m-%y', '%d.%m.%y', '%m/%d/%y', '%d-%b-%Y', '%d-%b-%y'):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    raise ValueError('no valid date format found')


def parse_date(elem):
    if(isinstance(elem,datetime)):
        return elem#.date()
    else:
        try:
            return try_parsing_date(elem.strip())#.date()
        except:
            return np.nan#datetime.now()#.date()


def parse_time(elem):

    if isinstance(elem, dtime):
        return int(elem.hour)

    time = elem.strip().replace(" ", "")

    try:
    #TODO: quelli com AM-PM vengono tradotti nella corrispondente ora in 24 ore??
        if re.compile(regex1).match(time) is not None:
            # 21:09:06-05:00
            return int(datetime.strptime(time, '%H:%M:%S-05:00').time().hour)

        if re.compile(regex2).match(time) is not None:
            # PM 7:59:16
            return int(datetime.strptime(time, '%p%I:%M:%S').time().hour)

        if re.compile(regex3).match(time) is not None:
            # 9:01:04 PM / 9:03:25 AM
            return int(datetime.strptime(time, '%I:%M:%S%p').time().hour)

        if re.compile(regex4).match(time) is not None:
            # 23:53:46
            return int(datetime.strptime(time, '%H:%M:%S').time().hour)

    except: return np.nan

    return np.nan

def translate_time(elem):
    if elem < 12:
        return 0
    if elem < 19:
        return 1
    return 2

