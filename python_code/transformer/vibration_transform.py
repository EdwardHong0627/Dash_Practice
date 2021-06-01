import pandas as pd
import pywt
import matplotlib.pyplot as plt
import os
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import hilbert
import numpy as np
import paho.mqtt.client as mqtt
from scipy.signal import stft
import json


class VibrationTransformation(object):

    def __init__(self, df: pd.DataFrame, config, draw=False, to_mqtt=True):
        self.df = df
        self.info_col = config["info_col"]
        self.draw = draw
        self.to_mqtt = to_mqtt
        self.config = config
        self.client = mqtt.Client(transport='websockets')
        # 設定登入帳號密碼
        self.client.username_pw_set(self.config['user'], self.config['password'])

        # 設定連線資訊(IP, Port, 連線時間)
        self.client.connect(self.config['mqtt_url'], self.config['mqtt_port'], 60)

    def time_domain(self, file_path: str, cols: list, window=10):
        info_data = self.df[self.info_col].copy()
        raw_data = self.df[cols].copy()
        statis_data = raw_data.rolling(window=window).agg(['mean', 'std', 'min', 'max',
                                                           'skew', 'kurt', self.rms, self.ptp])
        statis_data.columns = [i[1] + '_' + i[0] for i in statis_data.columns]
        statis_data = statis_data.dropna()
        all_data = pd.concat([info_data, statis_data], axis=1)
        all_data = all_data.dropna()
        all_data.reset_index(inplace=True)
        self.__send_mqtt(all_data, 'vibration/time_domain')
        return all_data


    def fft_and_fig(self, file_path: str, cols: list, N=None, T=None, draw=False):
        if N is None:
            N = len(self.df)
        print(N)
        if T is None:
            T = 60.0 / N
        fft_series = self.df.iloc[0][self.info_col]
        # fft_series['pass_number'] = fft_series['pass_number'].astype(str)
        # print(fft_series[self.info_col].values)
        # file_name = '_'.join(fft_series[self.info_col].values)
        # print(file_name)
        # if draw:
        #     fig, ax = plt.subplots(len(cols), 1, figsize=(16, 16), sharex=True, sharey=True)

        for idx, val in enumerate(cols):
            signal = self.df[val].to_numpy()
            fft_x = fftshift(fft(signal))
            fft_x_list = abs(fft_x).round(3).tolist()
            fft_series[val] = fft_x_list
            xf = fftshift(fftfreq(N, T))
            fft_series[val + '_ticks'] = xf.tolist()
        #     if draw:
        #         ax[idx].plot(xf, abs(fft_x))
        #         ax[idx].set_title(val)
        #         ax[idx].set_ylabel('magnitude')
        #         ax[idx].set_xlabel('Frequency [Hz]')
        # if draw:
        #     plt.savefig(os.path.join(file_path, file_name + '.png'))
        # fft_series.to_pickle(os.path.join(file_path, file_name + ".pkl"))
        self.__send_mqtt(fft_series, 'vibration/fft')
        return fftfreq, fft_series
    def wavelet_and_fig(self, file_path: str, cols: list, draw=False, level=1):
        dwt_series = self.df.iloc[0][self.info_col]
        fig_names = ['cA1']
        fig_names.extend(['cD' + str(i) for i in range(4, 0, -1)])
        # # dwt_series['pass_number'] = dwt_series['pass_number'].astype(str)
        # file_name = '_'.join(dwt_series[self.info_col].values)
        # if draw:
        #     fig, ax = plt.subplots(len(cols), level + 1, figsize=(64, 25), sharex=True, sharey=True)
        #     plt.xticks(fontsize=20)
        #     plt.yticks(fontsize=20)
        for key, val in enumerate(cols):
            signal = self.df[val].to_numpy()
            result = pywt.wavedec(signal, 'db1', level=level)
            for i, name in enumerate(fig_names[:level + 1]):
                a = result[i].tolist()
                dwt_series[val + '_' + name] = a
        #         if draw:
        #             ax[key, i].plot(a)
        #             ax[key, i].set_title(name)
        #             ax[key, i].tick_params(axis='x', labelsize=32)
        #             ax[key, i].tick_params(axis='y', labelsize=32)
        # if draw:
        #     plt.savefig(os.path.join(file_path, file_name + '.png'))
        # dwt_series.to_pickle(os.path.join(file_path, file_name + ".pkl"))
        self.__send_mqtt(dwt_series, 'vibration/dwt')
        return dwt_series
    def envelope_and_fig(self, file_path: str, cols: list, draw=False, freq=4096):
        envelope_series = self.df.iloc[0][self.info_col]
        # fig_names = ['cA1']
        # fig_names.extend(['cD' + str(i) for i in range(4, 0, -1)])
        # envelope_series['pass_number'] = envelope_series['pass_number'].astype(str)
        # file_name = '_'.join(envelope_series[self.info_col].values)

        # if draw:
        #     fig, ax = plt.subplots(len(cols), 1, figsize=(64, 25), sharex=True, sharey=True)
        #     plt.xticks(fontsize=20)
        #     plt.yticks(fontsize=20)
        for key, val in enumerate(cols):
            signal = self.df[val].to_numpy()
            m_x = fft(signal)
            fft_x = np.abs(m_x)[:int(freq / 2)]
            fft_x_tranf = hilbert(fft_x)
            amplitude_envelope = np.sqrt(np.array(fft_x) ** 2 + fft_x_tranf ** 2)
            envelope_series[val] = np.abs(amplitude_envelope)
        #     if draw:
        #         #             ax[key].plot(signal)
        #         ax[key].plot(amplitude_envelope, label='envelope')
        #         ax[key].set_title(val)
        #         ax[key].tick_params(axis='x', labelsize=32)
        #         ax[key].tick_params(axis='y', labelsize=32)
        # #     plt.show()
        # if draw:
        #     plt.savefig(os.path.join(file_path, file_name + '.png'))
        # print(envelope_series.transpose())
        # envelope_series.to_pickle(os.path.join(file_path, file_name + ".pkl"))
        self.__send_mqtt(envelope_series.transpose(), 'vibration/envelope')
        return envelope_series
    def stft_and_fig(self, file_path: str, cols: list, draw=False, fs=1e4, nperseg=1000):
        print(self.df.iloc[0])
        stft_series = self.df.iloc[0][self.info_col]
        # stft_series['pass_number'] = stft_series['pass_number'].astype(str)
        cm = plt.cm.get_cmap('rainbow')
        #file_name = '_'.join(stft_series[self.info_col].values)
        # if draw:
        #     fig, ax = plt.subplots(len(cols), 1, figsize=(16, 16))
        #     fig.tight_layout()
        for idx, val in enumerate(cols):
            signal = self.df[val].to_numpy()
            f, t, Zxx = stft(signal, fs, nperseg=nperseg)
            stft_series[val + '_' + 't'] = t
            stft_series[val + '_' + 'f'] = f
            stft_series[val + '_' + 'Zxx'] = np.abs(Zxx)
            if draw:
                ax[idx].pcolormesh(t, f, np.abs(Zxx), shading='gouraud', cmap=cm)
                ax[idx].set_title(val)
                plt.xlabel("Time [sec]")
                plt.ylabel("Frequency [Hz]")
        # if draw:
        #     plt.savefig(os.path.join(file_path, file_name + '.png'))
        # # stft_series.to_pickle(os.path.join(file_path, file_name + ".pkl"))
        print(stft_series.head())
        self.__send_mqtt(stft_series, 'vibration/stft')
        return stft_series
    def rms(self, x):
        return np.sqrt(np.mean(np.power(x.values, 2)))

    def ptp(self, x):
        return np.ptp(x.values)

    def __send_mqtt(self, result, topic):
        if self.to_mqtt:
            x = result.to_json(orient='index')
            print(x)
            self.client.publish(topic, json.dumps(x))