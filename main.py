# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from math import radians
import json
import scipy
import requests
import datetime
import os

sns.set_theme(style="whitegrid")


def get_data_from_source(url, column_name):
    raw_data = requests.get(url).json()
    raw_data_dataframe = pd.json_normalize(raw_data['results'][0]['series'], 'values')
    raw_data_dataframe.columns = ['time', column_name]
    raw_data_dataframe.set_index('time', drop=True, inplace=True)
    return raw_data_dataframe


def graph(ax, min_wind_speed: float, max_wind_speed: float, storage: list, max_expected_boat_speed: int):
    """
    Makes a single polar graph, which will be composed in the 4*4 grid
    """
    # Visual stuff
    ax.set_title(f"TWS {min_wind_speed}-{max_wind_speed}")
    ax.set_theta_zero_location('N')
    ax.set_ylim(0, max_expected_boat_speed)
    labels_list = ['0kn', '5kn', '10kn', '15kn', '20kn', '25kn', '30kn', '35kn', '40kn', '45kn', '50kn']
    ax.set_yticks(np.linspace(0, max_expected_boat_speed, max_expected_boat_speed // 5 + 1, endpoint=True))
    ax.set_yticklabels(labels_list[:max_expected_boat_speed // 5 + 1])
    ax.set_xticks(np.linspace(0, np.pi, 5, endpoint=True))
    ax.set_theta_direction(-1)
    ax.set_thetamin(0)
    ax.set_thetamax(180)

    # Consider only points between min_wind_speed and max_wind_speed
    current_wind_df = df[(df['TWS'] > min_wind_speed) & (df['TWS'] <= max_wind_speed)]
    # manual corrections to single graphs
    if max_wind_speed <= 7.5:
        current_wind_df = current_wind_df[(current_wind_df['BSP'] < 11)]
    if min_wind_speed == 5:
        current_wind_df = current_wind_df[(current_wind_df['adjTWA'] <= 160)]
    if min_wind_speed == 17.5:
        current_wind_df = current_wind_df[(current_wind_df['adjTWA'] <= 170)]

    x = [radians(x) for x in current_wind_df['adjTWA'].values]
    y = current_wind_df['BSP'].values
    # Scatter of all the points
    sns.scatterplot(x=x, y=y, s=9, ax=ax, hue=current_wind_df['BSP'].values, hue_norm=(0, 30))

    # define polynomial to optimize
    def func(angles, a, b, c, d, e, f):
        return np.poly1d([a, b, c, d, e, f])(angles)

    if not x:
        # if there are no wind values for this slice, just return
        return
    popt, pcov = scipy.optimize.curve_fit(func, current_wind_df['adjTWA'].values, current_wind_df['BSP'].values,
                                          maxfev=10000, bounds=(
            [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0], [np.inf, np.inf, np.inf, np.inf, np.inf, 0.01]))
    # Compute coefficents for best fit (all points). Boatspeed @ 0° is costrained at 0.
    ax.plot([radians(x) for x in current_wind_df['adjTWA'].values], func(current_wind_df['adjTWA'].values, *popt), '-r',
            linewidth=3, zorder=3)

    # Group the boatspeed values by TWA, choose 0.95 quartile
    grouped_wind_df = current_wind_df.groupby('adjTWA').quantile(0.95, interpolation='higher')

    # define polynomial2 to optimize. Fit it, boatspeed @0°=0
    def func1(angles, a, b, c, d, e, f, g):
        return np.poly1d([a, b, c, d, e, f, g])(angles)

    popt1, pcov1 = scipy.optimize.curve_fit(func1, [radians(x) for x in grouped_wind_df.index.values],
                                            grouped_wind_df['BSP'].values,
                                            maxfev=10000, bounds=(
            [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0],
            [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 0.01]))
    ax.plot([radians(x) for x in grouped_wind_df.index.values],
            func1([radians(x) for x in grouped_wind_df.index.values], *popt1), '--g',
            linewidth=3, zorder=2)

    storage.append({"par": popt1, "label": f"{min_wind_speed}-{max_wind_speed}"})
    # compute max VMG upwind and downwind, and plot it as 2 points on graph.

    vmg = func1([radians(x) for x in grouped_wind_df.index.values], *popt1) * np.cos(
        [radians(x) for x in grouped_wind_df.index.values])
    upwind_vmg = max(vmg)
    upwind_speed = func1([radians(x) for x in grouped_wind_df.index.values], *popt1)[
        int(np.where(vmg == upwind_vmg)[0])]
    upwind_vmg_direction = grouped_wind_df.index.values[int(np.where(vmg == upwind_vmg)[0])]
    upwind_vmg_direction_rad = [radians(x) for x in grouped_wind_df.index.values][int(np.where(vmg == upwind_vmg)[0])]

    downwind_vmg = min(vmg)
    downwind_speed = func1([radians(x) for x in grouped_wind_df.index.values], *popt1)[
        int(np.where(vmg == downwind_vmg)[0])]
    downwind_vmg_direction = grouped_wind_df.index.values[int(np.where(vmg == downwind_vmg)[0])]
    downwind_vmg_direction_rad = [radians(x) for x in grouped_wind_df.index.values][
        int(np.where(vmg == downwind_vmg)[0])]

    sns.scatterplot(x=[upwind_vmg_direction_rad, downwind_vmg_direction_rad], y=[upwind_speed, downwind_speed], s=50,
                    ax=ax, color='b', zorder=1, markers="o")
    ax.text(upwind_vmg_direction_rad, upwind_speed + 3, f'{round(upwind_speed, 2)}kn {int(upwind_vmg_direction)}°')
    ax.text(downwind_vmg_direction_rad, downwind_speed + 5,
            f'{round(downwind_speed, 2)}kn {int(downwind_vmg_direction)}°')
    ax.legend(title='BoatSpeed')


def generate_polars_image(storage):
    fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8),
          (ax9, ax10, ax11, ax12)) = plt.subplots(3, 4, figsize=(25, 25),
                                                                            subplot_kw=dict(projection='polar'))
    graph(ax1, 0, 2.5, storage, 15)
    graph(ax2, 2.5, 5, storage, 15)
    graph(ax3, 5, 7.5, storage, 15)
    graph(ax4, 7.5, 10, storage, 15)
    graph(ax5, 10, 12.5, storage, 30)
    graph(ax6, 12.5, 15, storage, 30)
    graph(ax7, 15, 17.5, storage, 30)
    graph(ax8, 17.5, 20, storage, 30)
    graph(ax9, 20, 22.5, storage, 30)
    graph(ax10, 22.5, 25, storage, 30)
    graph(ax11, 25, 27.5, storage, 30)
    graph(ax12, 27.5, 30, storage, 30)
    #graph(ax13, 30, 32.5, storage, 30)
    #graph(ax14, 32.5, 30, storage, 30)
    #graph(ax15, 30, 37.5, storage, 30)
    #graph(ax16, 37.5, 40, storage, 30)
    return fig


def generate_polar_file(parameters):
    def func1(angles, a, b, c, d, e, f, g):
        return np.poly1d([a, b, c, d, e, f, g])(angles)

    dictionary = {}
    angles = np.linspace(np.pi*(30/180), np.pi, 151)
    dictionary['angles'] = np.linspace(30, 180, 151)
    for element in parameters:
        dictionary[element["label"]] = [round(func1(angle, *element["par"]), 1) for angle in angles]
    return pd.DataFrame.from_dict(dictionary)


# SOG_URL = 'https://exocet.cloud/grafana/api/datasources/proxy/12/query?db=TeamMalizia&q=SELECT%20mean(
# %221s_ES.GPS_SOG%22)%20FROM%20%22Malizia_1s%22GROUP%20BY%20time(5s)%20fill(null)&epoch=ms' TWA_URL =
# 'https://exocet.cloud/grafana/api/datasources/proxy/12/query?db=TeamMalizia&q=SELECT%20mean(
# %221s_WTP_rename.TWA%22)%20FROM%20%22Malizia_1s%22GROUP%20BY%20time(5s)%20fill(null)&epoch=ms' TWS_URL =
# 'https://exocet.cloud/grafana/api/datasources/proxy/12/query?db=TeamMalizia&q=SELECT%20mean(
# %221s_WTP_rename.TWS%22)%20FROM%20%22Malizia_1s%22GROUP%20BY%20time(5s)%20fill(null)&epoch=ms'
SOG_URL = None
TWA_URL = None
TWS_URL = None

if not os.path.isfile('data.json'):
    open('data.json', 'w+').write('{"time":0}')

with open('data.json') as json_file:
    datafile = json.load(json_file)

if datafile.get('time') < int(datetime.datetime.timestamp(datetime.datetime.now()) - 3600):
    data = {"time": int(datetime.datetime.timestamp(datetime.datetime.now()))}
    speed = get_data_from_source(SOG_URL, 'BSP')
    TWA = get_data_from_source(TWA_URL, 'TWA')
    TWS = get_data_from_source(TWS_URL, 'TWS')
    df = speed.merge(TWA, how='left', left_index=True, right_index=True).merge(TWS, how='left', left_index=True,
                                                                               right_index=True)
    df = df.dropna(axis=0, how='any')
    df.index = df.index.values.astype(dtype='datetime64[ms]')
    df['adjTWA'] = df.apply(lambda row: abs(row['TWA']), axis=1)
    df = df.resample('0.5T').mean()
    df = df.sort_values('adjTWA')
    df = df.dropna(axis=0, how='any')
    df.adjTWA = df.adjTWA.round(0)
    df.TWS = df.TWS.round(1)
    df.BSP = df.BSP.round(1)
    df = df[(df['BSP'] > 0.5) & (df['BSP'] < 50)]
    df.to_pickle('dataframe.pkl')
    with open('data.json', 'w') as outfile:
        json.dump(data, outfile)
else:
    df = pd.read_pickle('dataframe.pkl')
    df = df[(df['BSP'] > 0.5) & (df['BSP'] < 50)]

parameter_storage = []
polars_png = generate_polars_image(parameter_storage)
polar_file=generate_polar_file(parameter_storage)
polar_file.to_excel('polars.xlsx')
polars_png.savefig('polars.png')
pass
# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#    pass

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
