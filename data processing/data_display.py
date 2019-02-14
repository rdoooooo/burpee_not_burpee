

from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from data_burpee_parser import parse_data
import time
from general_tools import butter_highpass_filter, butter_lowpass_filter
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumtrapz
from data_feature_extraction import filter_lowpass_single


def integrate(X):
    #velocity = cumtrapz(filter_lowpass_single(x), initial=0)
    velocity = cumtrapz(X, initial=0)
    displacement = cumtrapz(velocity)
    return velocity, displacement


if __name__ == '__main__':
    file = 'burpee_4.csv'
    df_parse, df = parse_data(loc=file)
    df_accel = df_parse
    #_, df_velocity, df_displacement = time_histories(df_parse)
df = df_accel.drop(df_accel.filter(regex='10').columns, axis=1)
df.head()
# df.to_pickle('df_vis.pk')
for i in range(int(len(df.columns) / 3)):
    if (i != 0) and (i != 5):
        continue

    # time.sleep(1)
    # plt.figure()
    # x = cumtrapz(df_accel['X_'+str(i)].iloc[200:])
    # z = cumtrapz(df_accel['Z_'+str(i)].iloc[200:])
    x_accel = df['X_' + str(i)].dropna().iloc[:]
    z_accel = df['Z_' + str(i)].dropna().iloc[:] * -1
    x_vel, x_disp = integrate(x_accel)
    z_vel, z_disp = integrate(z_accel)
    # x = cumtrapz(filter_lowpass_single(),initial=0)
    # z = cumtrapz(filter_lowpass_single(-1 *
    #                                    df_accel['Z_' + str(i)].dropna().iloc[300:]),initial=0)
    fig1 = plt.figure(0, figsize=(10, 10))
    if i < 5:
        plt.scatter(x_vel, z_vel, c='b')
    else:
        plt.scatter(x_vel, z_vel, c='r')
    fig2 = plt.figure(1, figsize=(10, 10))
    if i < 5:
        plt.scatter(x_disp, z_disp, c='b')
    else:
        plt.scatter(x_disp, z_disp, c='r')
    # plt.figure(1)
    # if i < 5:
    #     plt.scatter(x_disp, z_disp, c='b')
    # else:
    #     plt.scatter(x_disp, z_disp, c='r')


x = np.arange(10)
y = np.random.random(10)

fig = plt.figure()
plt.xlim(0, 10)
plt.ylim(0, 1)
graph, = plt.plot([], [], 'o')


def animate(i):
    graph.set_data(x[:i + 1], y[:i + 1])
    return graph


ani = FuncAnimation(fig, animate, frames=10, interval=200)
plt.show()


plt.plot(cumtrapz(x_accel))
z_accel.plot()
plt.plot(x_accel, z_accel)

plt.plot(x_vel, z_vel)
plt.plot(x_disp, z_disp)
plt.plot(z_disp)
plt.plot(x)
plt.figure(figsize=(10, 10))
for i in range(int(len(df_accel.columns) / 3)):
    # plt.figure()
    # x = cumtrapz(df_accel['X_'+str(i)].iloc[200:])
    # z = cumtrapz(df_accel['Z_'+str(i)].iloc[200:])
    x = cumtrapz(cumtrapz(filter_lowpass_single(-1 *
                                                df_accel['X_' + str(i)].dropna().iloc[:])))
    y = cumtrapz(cumtrapz(filter_lowpass_single(
        df_accel['Y_' + str(i)].dropna().iloc[:])))
    z = cumtrapz(cumtrapz(filter_lowpass_single(
        df_accel['Z_' + str(i)].dropna().iloc[:])))
    if i < 5:
        plt.scatter(x, z, c='b')
    else:
        plt.scatter(x, z, c='r')


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs=x, ys=y, zs=z, zdir='x', s=20)
ax.view_init()
for ii in range(0, 360, 1):
    ax.view_init(elev=10., azim=ii)
    plt.pause(1)
# for angle in range(0,6):
#     ax.view_init(elev = angle*30,azim = angle );
#     plt.draw();
#     plt.pause(1);
