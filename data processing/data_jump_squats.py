'''
Feature extraction for jumping squats
When doing the movement, there are periodic signals in the X direction
Doing a low pass filter for 1-2 Hz will extract the clean signal.
Doing a voting scheme for peaks in the X where a positive peak should
have 2 negative neighboring peaks. Build a window around positve peak and see
if there is a negative peak to the right and left.

Once there is confidence of the clean data and peaks, do a count
'''


import numpy as np
import pandas as pd
import peakutils
from data_reader_cleaner import load_data, filter_bandpass, filter_lowpass
import matplotlib.pyplot as plt
from sklearn.neighbors.nearest_centroid import NearestCentroid


def find_peaks_time(df, thres=0.12, min_dist=100, thres_abs=True, plots=0):
    df_peaks = pd.DataFrame()
    for col in df.columns:
        if 'filt' in col:
            data = df[col].values
            indexes = peakutils.indexes(
                data, thres=thres, min_dist=min_dist, thres_abs=True)
        #
            df_peaks[col] = pd.Series(data=indexes)
            if plots == 1:
                plt.figure(figsize=(15, 12))
                plt.plot(df.index, data)
                plt.plot(df[col].iloc[indexes].index,
                         np.abs(df[col].iloc[indexes].values))
                plt.title(col)
                plt.xlabel('Time')
                plt.ylabel('G')

    return df_peaks


def find_all_peaks(df_filt):
    df_filt_x = df_filt.filter(regex='X_filt_bp')
    df_peaks_pos = find_peaks_time(
        df_filt_x, thres=1, min_dist=25, thres_abs=True, plots=0)
    df_peaks_neg = find_peaks_time(
        df_filt_x * -1, thres=1, min_dist=25, thres_abs=True, plots=0)
    df_peaks = find_anomaly(df_peaks_pos, df_peaks_neg)
    return df_peaks


def count_jump_squats(df_peaks, max_window_size=30):
    jump_squat_count = 0
    pos_peaks = df_peaks[df_peaks.Dir == 1]
    for index in pos_peaks.index:
        print(index, jump_squat_count)
        # Does another check to see if leading point was a negative peak
        try:
            if df_peaks.iloc[index - 1].Dir != -1 or df_peaks.iloc[index + 1].Dir != -1:
                print(f'Check anomaly detector at index = {index}')
                continue
        except IndexError:
            print('Found last point - not included in number of peaks')
            continue
        # Subtracts neighboring neg peaks to see if they are within a specific window
        try:
            upper = abs(
                df_peaks.iloc[index - 1].X_filt_bp - df_peaks.iloc[index].X_filt_bp)
            lower = abs(
                df_peaks.iloc[index + 1].X_filt_bp - df_peaks.iloc[index].X_filt_bp)
        except IndexError:
            print('Found last point - not included in number of peaks')
            continue
        if upper < max_window_size and upper < max_window_size:
            jump_squat_count += 1

    return jump_squat_count


def find_anomaly(df_peaks_pos, df_peaks_neg):
    df_peaks_pos['Dir'] = 1
    df_peaks_neg['Dir'] = -1
    df_peaks = pd.concat([df_peaks_pos, df_peaks_neg], axis=0)
    df_binary = df_peaks.sort_values('X_filt_bp').reset_index(drop=True)

    index = True
    while index != False:
        index = valid(df_binary)
        # print(index)

        if index == None or index == False:
            continue

        df_binary.drop(labels=index, axis=0, inplace=True)
        df_binary.reset_index(drop=True, inplace=True)
    return df_binary


def valid(df_peaks, pattern=np.array([-1, 1, -1])):
    data = df_peaks.Dir
    for i in range(len(data)):
        if data[i] != pattern[i % 3]:
            return i
    return False


if __name__ == '__main__':
    # Load data and filter
    df = load_data(loc='jumping_jacks_1.csv')
    #df = load_data(loc='jump_squats_1.csv')
    df = load_data(loc='burpee_1.csv')
    df = load_data(loc='test_1.csv')
    df_filt = filter_bandpass(df, lowcut=1, highcut=5, fs=100)
    #df_filt_hp = filter_lowpass(df,  lowcut=10, fs=100)
    df_peaks = find_all_peaks(df_filt)

    jump_squats = count_jump_squats(df_peaks, max_window_size=30)
    print(f'Found {jump_squats} jump squats!')

# Plot data that shows the visual separation
df.X_filt_bp.iloc[:2000].plot(figsize=(10, 10))
df_filt.Y_filt_bp.iloc[:].plot(figsize=(10, 10))
df_filt_hp.X_filt_hp.iloc[:].plot(figsize=(10, 10))
df_filt_hp.Y_filt_hp.iloc[:].plot(figsize=(10, 10))
