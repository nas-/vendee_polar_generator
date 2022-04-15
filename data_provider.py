import pandas as pd
import json
import os
import datetime
import requests
import numpy as np

SOG_URL = 'https://exocet.cloud/grafana/api/datasources/proxy/12/query?db=TeamMalizia&q=SELECT%20mean( %221s_ES.GPS_SOG%22)%20FROM%20%22Malizia_1s%22GROUP%20BY%20time(5s)%20fill(null)&epoch=ms'
TWA_URL = 'https://exocet.cloud/grafana/api/datasources/proxy/12/query?db=TeamMalizia&q=SELECT%20mean(%221s_WTP_rename.TWA%22)%20FROM%20%22Malizia_1s%22GROUP%20BY%20time(5s)%20fill(null)&epoch=ms'
TWS_URL = 'https://exocet.cloud/grafana/api/datasources/proxy/12/query?db=TeamMalizia&q=SELECT%20mean(%221s_WTP_rename.TWS%22)%20FROM%20%22Malizia_1s%22GROUP%20BY%20time(5s)%20fill(null)&epoch=ms'


class DataProvider(object):
    def __init__(self, force=None):
        self._last_run = self._get_last_run()
        if (self._last_run < datetime.datetime.now().timestamp() - 3600) or force:
            self._refresh_data()
        self.data = self._read_data()

    def _get_last_run(self):
        if not os.path.isfile('data.json'):
            open('data.json', 'w+').write('{"time":0}')
            return 0

        with open('data.json') as json_file:
            content = json.load(json_file)
        return content.get('time')

    def _refresh_data(self):

        data = {"time": int(datetime.datetime.timestamp(datetime.datetime.now()))}
        print('fetching SOG')
        speed = self._get_data_from_source(SOG_URL, 'BSP')
        print('fetching TWA')
        TWA = self._get_data_from_source(TWA_URL, 'TWA')
        print('fetching TWS')
        TWS = self._get_data_from_source(TWS_URL, 'TWS')
        df = speed.merge(TWA, how='left', left_index=True, right_index=True).merge(TWS, how='left', left_index=True,
                                                                                   right_index=True)
        df = df.dropna(axis=0, how='any')
        df.index = df.index.values.astype(dtype='datetime64[ms]')
        df['adjTWA'] = np.abs(df['TWA'])
        df = df.sort_values('adjTWA')
        df.to_pickle('dataframe.pkl')
        print('saved data to disk')
        with open('data.json', 'w') as outfile:
            json.dump(data, outfile)

    def _read_data(self):
        df = pd.read_pickle('dataframe.pkl')
        df = df[(df['BSP'] > 0.5) & (df['BSP'] < 50)]
        return df

    def _get_data_from_source(self, url, name):
        raw_data = requests.get(url).json()
        raw_data_dataframe = pd.json_normalize(raw_data['results'][0]['series'], 'values')
        raw_data_dataframe.columns = ['time', name]
        raw_data_dataframe.set_index('time', drop=True, inplace=True)
        return raw_data_dataframe


# if not os.path.isfile('data.json'):
#     open('data.json', 'w+').write('{"time":0}')
#
# with open('data.json') as json_file:
#     datafile = json.load(json_file)
#
# if datafile.get('time') < int(datetime.datetime.timestamp(datetime.datetime.now()) - 3600):
#     data = {"time": int(datetime.datetime.timestamp(datetime.datetime.now()))}
#     speed = get_data_from_source(SOG_URL, 'BSP')
#     TWA = get_data_from_source(TWA_URL, 'TWA')
#     TWS = get_data_from_source(TWS_URL, 'TWS')
#     df = speed.merge(TWA, how='left', left_index=True, right_index=True).merge(TWS, how='left', left_index=True,
#                                                                                right_index=True)
#     df = df.dropna(axis=0, how='any')
#     df.index = df.index.values.astype(dtype='datetime64[ms]')
#     df['adjTWA'] = df.apply(lambda row: abs(row['TWA']), axis=1)
#     df = df.resample('0.5T').mean()
#     df = df.sort_values('adjTWA')
#     df = df.dropna(axis=0, how='any')
#     df.adjTWA = df.adjTWA.round(0)
#     df.TWS = df.TWS.round(1)
#     df.BSP = df.BSP.round(1)
#     df = df[(df['BSP'] > 0.5) & (df['BSP'] < 50)]
#     df.to_pickle('dataframe.pkl')
#     with open('data.json', 'w') as outfile:
#         json.dump(data, outfile)
# else:
#     df = pd.read_pickle('dataframe.pkl')
#     df = df[(df['BSP'] > 0.5) & (df['BSP'] < 50)]

if __name__ == '__main__':
    data = DataProvider(force=True)
