import numpy as np
import pandas as pd
import calendar

def set_time_of_day(n):
    if n <= 7:
        return "night"
    elif n <= 23:
        return "morning"
    elif n <= 39:
        return "afternoon"
    else:
        return "night"

if __name__ == '__main__':
    # enumeration of the 2,880 time increments
    increments = np.arange(2880)
    # assign a day number (0 to 59) for all time increments
    day_nums = increments/48
    # assign an integer (0 to 6) for all time increments so that the first day is Friday
    day_indices = (4 + day_nums)%7
    # convert the day indices into strings abbreviating the name of the days
    day_names = np.array( [list(calendar.day_abbr)[n] for n in day_indices] )
    # boolean vector with 1 values if the time increment corresponds to a weekend day
    weekend = np.array( [1 if day in ["Sat", "Sun"] else 0 for day in day_names] )
    time_of_day = np.array( [set_time_of_day(n) for n in increments%48] )
