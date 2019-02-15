from scipy.signal import hilbert
from scipy import signal
import scipy as sp
from scipy.signal import find_peaks_cwt
import numpy as np
from general_tools import butter_highpass_filter, butter_lowpass_filter, butter_bandpass_filter
import pandas as pd
import matplotlib.pyplot as plt
import peakutils


def load_data(loc='burpee_1.csv'):
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
        # plt.subplot(len(df.columns), 1, i+1)
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
        # plt.subplot(len(df.columns), 1, i+1)
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


def filter_lowpass(df, lowcut=5, fs=100):
    for col in df.columns:
        if 'filt' not in col:
            data = df[col].values
            data_filt = butter_lowpass_filter(data=data, lowcut=lowcut, fs=fs)
            df[col + '_filt_lp'] = pd.Series(data=data_filt, index=df.index)
    return df


def filter_bandpass(df, lowcut=1, highcut=5, fs=100):
    for col in df.columns:
        if 'filt' not in col:
            data = df[col].values
            data_filt = butter_bandpass_filter(
                data=data, lowcut=lowcut, highcut=highcut, fs=fs)
            df[col + '_filt_bp'] = pd.Series(data=data_filt, index=df.index)
    return df


def filter_highpass(df, highcut=40, fs=100):
    for col in df.columns:
        if 'filt' not in col:
            data = df[col].values
            data_filt = butter_highpass_filter(
                data=data, highcut=highcut, fs=fs)
            df[col + '_filt_hp'] = pd.Series(data=data_filt, index=df.index)
    return df


def find_peaks(df, thres=0.12, min_dist=100, thres_abs=True, plots=0):
    df_peaks = pd.DataFrame()
    for col in df.columns:
        if 'filt' in col:
            env = np.abs(hilbert(df[col].values))
            indexes = peakutils.indexes(
                env, thres=thres, min_dist=min_dist, thres_abs=True)
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

    return df_peaks


if __name__ == '__main__':
    # Load data
    df = load_data(loc='test_1.csv')
    # df = load_data(loc='jumping_jacks_1.csv')
    df = load_data(loc='walking_around.csv')

    # df.to_pickle('df_walking_around_all_sensors.pk')
    df = filter_highpass(df, highcut=30, fs=100)
    #df = filter_bandpass(df, lowcut=1, highcut=5, fs=100)
    # df = filter_lowpass(df)
    plot_filt(df)
    df_peaks = find_peaks(df, thres=0.2, min_dist=400, thres_abs=True, plots=1)

#
# df['Y_filt_hp'].iloc[:-500].plot(figsize=(10, 10))
# #
# df['Y_filt_bp'].iloc[100:2500].plot(figsize=(10, 10))
# #
# #
# df['Y'].iloc[600:1400].plot(figsize=(10, 10))
