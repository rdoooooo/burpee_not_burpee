
'''
Data has been parsed into individual burpees which yields N*3 columns per burpee
that has been detected. There will be a signal for each direction. Now
we would like to answer whether the user has jumped after doing the drop.
Hypothesis, if the user has jumped than there will be more motion in all directions

'''
from scipy.integrate import trapz
import pandas as pd
from data_burpee_parser import parse_data
import matplotlib.pyplot as plt
import importlib as imp
from scipy.integrate import cumtrapz
import numpy as np
from scipy.signal import detrend
from general_tools import butter_highpass_filter, butter_lowpass_filter


def max_peaks(df):
    df = filter_highpass(df)
    df_peak = pd.DataFrame()
    for col in df.columns:
        df_peak[col] = pd.Series(df[col].iloc[:300].iloc[100:].abs().idxmax())
    return df_peak


def filter_highpass(df):
    for col in df.columns:
        if 'filt' not in col:
            data = df[col].fillna(0).values
            data_filt = butter_highpass_filter(data=data, highcut=5, fs=100)
            df[col] = pd.Series(data=data_filt, index=df.index)
    return df


def filter_highpass_single(data):
    data_filt = butter_highpass_filter(data=data, highcut=5, fs=100)
    return data_filt


def filter_lowpass_single(data, lowcut=25, fs=100):
    data_filt = butter_lowpass_filter(data=data, lowcut=25, fs=100)
    return data_filt


def plot_signals(df_parse):
    # Plot parse signals
    for col in df_parse.columns:
        # if 'X' in col:
        plt.figure()
        df_parse[col].plot()
        plt.title(col)


# col='Z_6'
# peak
# len(extracted_data)
# len(x_vec)


def feature_extraction(df_parse):
    # constant = 32.174  # g - > 32.2
    #df_parse_clean = filter_highpass(df_parse)
    df_max_peaks = max_peaks(df_parse)
    df_features = pd.DataFrame()

    for col in df_parse:
        peak = df_max_peaks[col].values[0]
        # Pulls out only data after the pushup
        extracted_data = df_parse[col].iloc[peak + 50:]

        # Max, AUC, std - acceleration
        max_accel = extracted_data.abs().max()
        energy_accel = np.sqrt(extracted_data.abs().pow(2).sum())
        std_accel = extracted_data.std()

        extracted_filt = filter_highpass_single(
            df_parse[col].iloc[peak + 50:].values)

        # Calculate velocity and displacement
        x_vec = np.arange(start=0, stop=(len(extracted_data)) *
                          1 / 100, step=1 / 100)

        # Adds a check to make sure x_vec and extracted_data are the same length
        if len(extracted_filt) != len(x_vec):
            print(col)
            print('Lengths of extracted_data does not match x_vec')
            print('x_vec length is being adjusted to match extracted_data')
            x_vec = x_vec[:len(extracted_filt)]

        # Calculate velocity by taking the integral of acceleration
        velocity = cumtrapz(y=extracted_filt, x=x_vec, initial=0.0)
        # High pass filter velocity curve to make sure it starts at zero
        vel_filt = filter_highpass_single(data=velocity)
        # Calculate displacement by taking integral of velocity
        displacement = cumtrapz(y=vel_filt, x=x_vec, initial=0.0)

        # Max, AUC, std - velocity
        max_velocity = np.max(np.abs(velocity))
        energy_velocity = np.sum(np.power(velocity, 2))
        std_velocity = np.std(velocity)

        # Max, AUC, std - displacement
        max_displacement = np.max(np.abs(displacement))
        energy_displacement = np.sum(np.power(displacement, 2))
        std_displacement = np.std(displacement)

        # Pulls out the f
        features = [max_accel, energy_accel, std_accel,
                    max_velocity, energy_velocity, std_velocity,
                    max_displacement, energy_displacement, std_displacement]

        feature_str = ['accel_max', 'accel_energy', 'accel_std',
                       'vel_max', 'vel_energy', 'vel_std',
                       'disp_max', 'disp_energy', 'disp_std']

        df_features[col] = pd.Series(data=features, index=feature_str)

    return df_features


def time_histories(df_parse):
    # constant = 32.174  # g - > 32.2
    df_parse_clean = filter_highpass(df_parse)
    #df_max_peaks = max_peaks(df_parse_clean)
    df_velocity = pd.DataFrame()
    df_displacement = pd.DataFrame()

    for col in df_parse_clean:
        # Calculate velocity and displacement
        x_vec = np.arange(start=0, stop=(len(df_parse_clean[col].values)) *
                          1 / 100, step=1 / 100)

        velocity = cumtrapz(y=df_parse_clean[col].values, x=x_vec, initial=0.0)
        vel_filt = filter_highpass_single(data=velocity)
        displacement = cumtrapz(y=vel_filt, x=x_vec, initial=0.0)

        df_velocity[col] = pd.Series(data=velocity, index=df_parse_clean.index)
        df_displacement[col] = pd.Series(
            data=displacement, index=df_parse_clean.index)

    return df_parse, df_velocity, df_displacement


def combine_directions(df_features, set):

    df_data = pd.DataFrame()
    for i in range(int(len(df_features.columns) / 3)):
        col_str = ['X_' + str(i), 'Y_' + str(i), 'Z_' + str(i)]
        # Grab the X Y and Z columns for each burpee
        X = df_features[col_str].filter(regex='X')
        X.index = X.index + '_X'

        Y = df_features[col_str].filter(regex='Y')
        Y.index = Y.index + '_Y'

        Z = df_features[col_str].filter(regex='Z')
        Z.index = X.index + '_Z'

        # Combines all index string for each direction together
        index_str = np.concatenate([X.index, Y.index, Z.index])

        # Combines all values for each direction together
        data = np.reshape(df_features[col_str].values, newshape=(
            len(df_features[col_str]) * 3, ))

        # Puts data and index into a dataframe
        df_data[str(set) + '_' + str(i)] = pd.Series(
            data=data,
            index=index_str
        )

    return df_data


def add_truth(df_data, truth_key):
    if len(truth_key) != len(df_data.columns):
        print('Check lens of the truth key and df_data')
        print(len(df_data.columns))
        return
    truth = pd.Series(data=truth_key, index=df_data.columns, name='Truth')
    df_data = df_data.append(truth)
    return df_data


def save_df(df_data, name, loc='data/'):
    df_data.to_pickle(path=loc + name)
    print('DF saved at ' + loc + name)


if __name__ == '__main__':
    file = 'burpee_test_1.csv'
    df_parse, df = parse_data(loc=file)
    df_features = feature_extraction(df_parse)
    df_data = combine_directions(df_features, set=1)
    #truth_key = np.ones(shape=(11,))
    truth_key = np.hstack([np.ones(shape=(10,)), np.zeros(shape=(10,))])
    #truth_key[0] = 0
    #truth_key = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    #truth_key = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    #truth_key = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    df_feat = add_truth(df_data, truth_key)
    save_df(df_feat, name=file[:-4] + '.pk')

    for col in df_parse.columns:
        if 'X' in col:
            plt.figure()
            plt.title(col)
            df_parse[col].plot()
