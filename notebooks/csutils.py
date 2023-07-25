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
        yield start_timestamp, dataframe.loc[start_timestamp:end_timestamp]
        


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


def get_histograms_from_window(window, density=False, way_of_binarization=None):
    all_hist = []
    all_bins = []
    all_dens = []
    for sensor in window.columns:
        column_data = window[sensor]
        if way_of_binarization is None:
            hist, bins = np.histogram(column_data, bins='auto')
            if density:
                dens = np.histogram(column_data,bins='auto',density=density)
        else:
            bins = way_of_binarization(column_data)
            hist, bins = np.histogram(column_data, bins=bins)
            if density:
                dens = np.histogram(column_data,bins=bins,density=density)
        all_hist.append(hist)
        all_bins.append(bins)
        all_dens.append(dens)
    if density:
        return all_bins, all_hist, all_dens
    return all_bins, all_hist

def get_time_series_from_window(window, start, window_size, AVG_SIZE='1S'):
    time_series = []
    window_cp = window.copy()
    has_start = start in window_cp.index
    has_end = start+window_size in window_cp.index
    for sensor in window_cp.columns:
        column_data = window_cp[sensor]
        if not has_start:
            column_data.loc[start] = np.nan
        if not has_end:
            column_data.loc[start+window_size] = np.nan
        averages = column_data.resample(AVG_SIZE).mean()
        np.nan_to_num(averages, copy=False)
        averages = averages.values
        time_series.append(averages)
    return time_series

