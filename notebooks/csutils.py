import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

def generate_window_iterator(start, end, freq):
    current_timestamp = start
    
    while current_timestamp + freq <= end:
        yield current_timestamp
        current_timestamp += freq

def extract_time_windows(dataframe, window_size, timestamp = False, **kwargs):

    window_iterator = generate_window_iterator(**kwargs)
    
    for start_timestamp in window_iterator:
        end_timestamp = start_timestamp + window_size
        if timestamp:
            yield (start_timestamp, dataframe.loc[start_timestamp:end_timestamp])
            continue
        yield dataframe.loc[start_timestamp:end_timestamp]
        


def get_numeric_from_window(window):
    measurements = []
    for sensor in window.columns:
        mean = window[sensor].mean()
        std = window[sensor].std()
        skewness = skew(window[sensor])
        kurt = kurtosis(window[sensor])
        quantiles = np.percentile(window[sensor], [0, 25, 50, 75, 100])
        deciles = np.percentile(window[sensor], range(10, 100, 10))
        # END OF CALCULATIONS
        sensor_measurements = np.concatenate(([mean, std, skewness, kurt], quantiles, deciles))
        measurements.append(sensor_measurements)
    return measurements


def get_histograms_from_window(window, way_of_binarization=None):
    histograms = []
    for sensor in window.columns:
        column_data = window[sensor]
        
        if way_of_binarization is None:
            hist, bins = np.histogram(column_data, bins='auto')
        else:
            bins = way_of_binarization(column_data)
            hist, bins = np.histogram(column_data, bins=bins)
            
        histograms.append((bins, hist))
    return histograms

def get_time_series_from_window(window, start, window_size, AVG_SIZE='1S'):
    time_series = []

    for sensor in window.columns:
        cp = window[sensor].copy()
        cp.loc[start] = np.nan
        cp.loc[start+window_size] = np.nan
        averages = cp.resample(AVG_SIZE).mean()
        np.nan_to_num(averages, copy=False)
        time_series.append(averages)

    return time_series

