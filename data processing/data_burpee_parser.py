'''
Find locations where all 3 axes have had a peak, this is a feature
of the burpee to have an psuedo-impluse in all 3 directions. Given an impulse
in the time domain we could take advantage of the frequency properties (broad
frequency response) to implement a high pass filter that will locate those impulses

1) high pass filter signals
2) detect peaks
3) check if peaks are within an X window of each other (confirms 3 directional peak)
4) use X axis as reference point
5) extract data based on a maximum time before and after impulses
6) return data that has been parsed - this represents how many potential
burpees have been detected
7) returns parsed data and filtered data
'''

from sklearn.neighbors.nearest_centroid import NearestCentroid
from scipy.signal import hilbert
from scipy import signal
import scipy as sp
from scipy.signal import find_peaks_cwt
import numpy as np
from general_tools import butter_highpass_filter, butter_lowpass_filter

import pandas as pd
import matplotlib.pyplot as plt
import peakutils
#!pwd


def load_data(loc):

    df = pd.read_csv('data/' + loc, sep='|')
    df['time'] = df["loggingTime(txt)"].apply(pd.to_datetime)
    df.set_index('time', inplace=True)
    df = down_select(df)
    return df


def plot_all(df):
    # print(accel_col)
    # plt.figure(figsize=(15,25))
    for i, col in enumerate(df.columns):
        plt.figure(figsize=(12, 10))
        #plt.subplot(len(df.columns), 1, i+1)
        df[col].plot()
        # plt.ylim([-5,5])
        plt.title(col)
        plt.show()


def plot_filt(df):
    labels = ['X', 'Y', 'Z']
    # print(accel_col)
    # plt.figure(figsize=(15,25))
    for axis in labels:
        # plt.figure(figsize=(15,12))
        #plt.subplot(len(df.columns), 1, i+1)
        df.filter(regex=axis).plot(figsize=(15, 12))
        # df[axis+'_filt'].plot()
        # plt.ylim([-5,5])
        plt.title(axis)
        plt.show()


def down_select(df):
    df_temp = df[[
        'motionUserAccelerationX(G)', 'motionUserAccelerationY(G)', 'motionUserAccelerationZ(G)']]
    df_accel = df_temp.rename(columns={'motionUserAccelerationX(G)': 'X',
                                       'motionUserAccelerationY(G)': 'Y', 'motionUserAccelerationZ(G)': 'Z'})
    return df_accel


def filter_lowpass(df):
    for col in df.columns:
        data = df[col].values
        data_filt = butter_lowpass_filter(data=data, lowcut=40, fs=100)
        df[col + '_filt_lp'] = pd.Series(data=data_filt, index=df.index)
    return df


def filter_highpass(df, highcut=40, fs=100):
    for col in df.columns:
        data = df[col].values
        data_filt = butter_highpass_filter(data=data, highcut=40, fs=100)
        df[col + '_filt_hp'] = pd.Series(data=data_filt, index=df.index)
    return df


def find_peaks(df, threshold=0.12, min_dist=100, thres_abs=True, plots=0):
    df_peaks = pd.DataFrame()
    for col in df.columns:
        if 'filt' in col:
            env = np.abs(hilbert(df[col].values))
            indexes = peakutils.indexes(
                env, thres=threshold, min_dist=min_dist, thres_abs=True)
        #
            df_peaks[col] = pd.Series(data=indexes)
            if plots == 1:
                plt.figure(figsize=(15, 12))
                plt.plot(df.index, env)
                plt.plot(df[col].iloc[indexes].index, env[indexes])
                # np.abs(df[col].iloc[indexes].values))
                plt.title(col)
                plt.xlabel('Time')
                plt.ylabel('G')

    if len(df_peaks) != 0:
        votes = vote_scheme(df_peaks)
        print('Found {} burpees'.format(sum(votes)))
        return df_peaks[votes]
    else:
        print('Found {} burpees'.format(0))
        return 0


def vote_scheme(df_peaks):
    # Apply a voting scheme using the nearest neighbors (NN) to see if peaks in the
    # other axes are near the one found for X. If a peak gets 3 votes than it gives
    # higher confidence that motion was
    n_peaks = [len(df_peaks[col].dropna()) for col in df_peaks.columns]

    # Will use the direction with the greatest number of peaks as the base
    # The other two direction will vote to see which peaks they have matching
    # If they all have the same number of peaks, default to X-axis
    if len(set(n_peaks)) == 1:
        base_dir = df_peaks.iloc[:, 0]
        voting_dirs = df_peaks.iloc[:, 1:]
    else:

        highest_n_peak = [peak == max(n_peaks) for peak in n_peaks]
        lower_n_peak = [peak != max(n_peaks) for peak in n_peaks]
        base_dir = df_peaks.X_filt_hp
        voting_dirs = df_peaks.loc[:, lower_n_peak]

    X = np.array(base_dir.values).reshape(-1, 1)
    y = np.array(base_dir.index.values)
    clf = NearestCentroid()
    clf.fit(X, y)
    NearestCentroid(metric='euclidean', shrink_threshold=None)
    total_votes = np.ones(len(base_dir))
    for col in voting_dirs:
        votes = clf.predict(
            np.array(voting_dirs[col].dropna().values).reshape(-1, 1))
        total_votes[votes] += 1
    peaks = (total_votes == len(df_peaks.columns))
    #print(peaks, votes)
    return peaks


def parser(df, df_peaks, window_size=700, adjustment=100):
    # For set 8
    # df_peaks.drop(axis=0, index=8, inplace=True)
    # df_peaks.iloc[6].X_filt_hp += 100
    # df_peaks.iloc[14].X_filt_hp += 150
    # For set 7
    #df_peaks.iloc[1].X_filt_hp += 50
    if type(df_peaks) == int:
        return 0
    print(df_peaks.shape)
    for i, peak in enumerate(df_peaks.X_filt_hp):
        # Determine window of time to grab
        start = int(peak - adjustment)
        stop = int(peak + window_size - adjustment)

        # Check if window exceeds the start of the next window,
        # if so then set the stop of the signal to be the start of the neighboring
        # signal minus a 150 steps
        # of

        if i + 1 != len(df_peaks):
            next_peak = df_peaks.X_filt_hp.values[i + 1]
            if stop + 50 > next_peak:
                stop = int(next_peak - adjustment)
        #print(start, stop)

        # Parse data from full dataframe
        try:
            df_temp = pd.DataFrame(data=df[['X', 'Y', 'Z']].iloc[start:stop].values,
                                   index=np.arange(stop - start) * 3,
                                   columns=['X_' + str(i), 'Y_' + str(i), 'Z_' + str(i)])
            #print(start, stop)
        except:
            try:
                print(i, start, stop)

                window = window_size  # 6 seconds
                start = peak
                stop = df.shape[0] - 150
                df_temp = pd.DataFrame(data=df[['X', 'Y', 'Z']].iloc[start:stop].values,
                                       index=np.arange(stop - start) * 3,
                                       columns=['X_' + str(i), 'Y_' + str(i), 'Z_' + str(i)])
                print(start, stop)
                print('Decreasing last window - not enough time after burpee')
            except:
                print(
                    f'Could not pull window between {start} and {stop} for window {i} peak{peak}')
                continue
        # Combine dataframes into a large one with all parsed sections
        try:
            df_parse = pd.concat([df_parse, df_temp], axis=1)
        except:
            df_parse = df_temp
    print('Extracted {} windows per direction'.format(len(df_parse.columns) / 3))
    return df_parse


def parse_data(loc):
    #global df
    df = load_data(loc=loc)
    df = filter_highpass(df, highcut=30, fs=100)
    # plot_filt(df)
    #df_peaks = find_peaks(threshold=.065, min_dist=350, plots=0)
    # df_peaks = find_peaks(df, threshold=.2, min_dist=100,
    #                       thres_abs=False, plots=0)
    df_peaks = find_peaks(df, threshold=0.07, min_dist=400,
                          thres_abs=True, plots=1)
    df_parse = parser(df, df_peaks)
    df_parse.index = np.arange(len(df_parse))
    return df_parse, df


loc = 'burpee_test_1.csv'
df_parse, df = parse_data(loc)
#
# for col in df.columns:
#     # if 'X' in col:
#     plt.figure()
#     df[col].plot()
#     plt.title(col)
#
# for col in df_parse.columns:
#     if 'X' in col:
#         plt.figure()
#         df_parse[col].plot()
#         plt.title(col)
# df['X'].iloc[150:1500].plot(figsize=(10, 10))
# df['Z'].iloc[100:2500].plot(figsize=(10, 10))
# df['Y'].iloc[200:1200].plot(figsize=(10, 10))
