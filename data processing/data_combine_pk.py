'''
Combines all pk into a single dataframe to be inputed into a classifier
Performs task for training and test set
'''

import pandas as pd
import os

# Training set
path = '/Users/richarddo/Documents/github/Metis/Projects/Project_3_Mcnulty/data/'
df = pd.DataFrame()
for file in os.listdir(path):
    if '.pk' in file and 'full' not in file and 'test' not in file:
        df_temp = pd.read_pickle(path + file)

        df = pd.concat([df, df_temp], axis=1)

df = df.reindex(sorted(df.columns), axis=1)
df = df.T
for col in df.columns:
    if 'X_Z' in col:
        df.rename({col: col[:-3] + 'Z'}, axis=1, inplace=True)

df.to_pickle(path='data/df_train.pk')
df = pd.read_pickle('data/df_train.pk')


# Test set
df_test = pd.DataFrame()
for file in os.listdir(path):
    if '.pk' in file and 'full' not in file and 'test' in file:
        df_temp = pd.read_pickle(path + file)

        df_test = pd.concat([df_test, df_temp], axis=1)

df_test = df_test.reindex(sorted(df_test.columns), axis=1)
df_test = df_test.T
for col in df_test.columns:
    if 'X_Z' in col:
        df_test.rename({col: col[:-3] + 'Z'}, axis=1, inplace=True)

df_test.to_pickle(path='data/df_test.pk')
df_test = pd.read_pickle('data/df_test.pk')
