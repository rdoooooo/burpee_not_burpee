'''
Feature extraction for jumping jacks
When doing the movement, there are periodic signals in the X and Y directions
Doing a low pass filter for 1-2 Hz will extract the clean signal.
Doing a voting scheme for peaks in the X and Y direction, where the peaks
must align with a small percentage will give it confidence that it is in fact
a jumping jack

Once there is confidence of the clean data and peaks, do a count above a
certain threshold.
'''


import numpy as np
import pandas as pd
import peakutils
from data_reader_cleaner import load_data, filter_bandpass
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
        base_dir = df_peaks.iloc[:, highest_n_peak]
        voting_dirs = df_peaks.iloc[:, lower_n_peak].dropna()
    if len(base_dir) == 0 or len(voting_dirs) == 0:
        df_peaks_voted = pd.DataFrame()
        return df_peaks_voted
    # Do a KNN to vote if both directions have the a peak near same
    # Allow for a 5% max fluxuation
    X = np.array(base_dir.values).reshape(-1, 1)
    y = np.array(base_dir.index.values)
    clf = NearestCentroid()
    clf.fit(X, y)
    NearestCentroid(metric='euclidean', shrink_threshold=None)
    total_votes = np.ones(len(base_dir))
    for col in voting_dirs:
        votes = clf.predict(
            np.array(voting_dirs[col].values).reshape(-1, 1))
        total_votes[votes] += 1

    peaks_votes = (total_votes == len(df_peaks.columns))
    # Check how much each row differs - max 5%
    df_base = base_dir[peaks_votes]
    df_base.reset_index(drop=True, inplace=True)
    df_peaks_temp = pd.concat([df_base, voting_dirs], axis=1)

    df_peaks_temp['10pt_diff'] = df_peaks_temp['X_filt_bp'].sub(
        df_peaks_temp['Y_filt_bp']).abs() < 10
    df_peaks_voted = df_peaks_temp[df_peaks_temp['10pt_diff']]
    df_peaks_voted.drop('10pt_diff', axis=1, inplace=True)
    return df_peaks_voted


def count_jumping_jacks(df_filt):
    df_filt_x, df_filt_y = df_filt.filter(
        regex='X_filt_bp'), df_filt.filter(regex='Y_filt_bp')
    df_peaks_x = find_peaks_time(
        df_filt_x, thres=.7, min_dist=25, thres_abs=True, plots=0)
    df_peaks_y = find_peaks_time(
        df_filt_y, thres=.7, min_dist=50, thres_abs=True, plots=0)
    df_peaks = pd.concat([df_peaks_x, df_peaks_y], axis=1)
    df_peaks_voted = vote_scheme(df_peaks)
    return df_peaks_voted.shape[0]


if __name__ == '__main__':
    # Load data and filter
    df = load_data(loc='jumping_jacks_1.csv')
    #df = load_data(loc='jump_squats_1.csv')
    #df = load_data(loc='burpee_1.csv')
    df = load_data(loc='test_1.csv')
    df_filt = filter_bandpass(df, lowcut=1, highcut=2, fs=100)
    jumping_jacks = count_jumping_jacks(df_filt)
    print(f'Found {jumping_jacks} jumping jacks!')

# Plot data that shows the visual separation
df_filt.X_filt_bp.iloc[:].plot(figsize=(10, 10))
df_filt.Y_filt_bp.iloc[:].plot(figsize=(10, 10))
