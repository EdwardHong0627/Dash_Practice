import pandas as pd
from scipy import signal
from pywt import cwt
import matplotlib.pyplot as plt
import os
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import hilbert
import numpy as np
import paho.mqtt.client as mqtt
from scipy.signal import stft
import json


class VibrationTransformation(object):

    def __init__(self, df: pd.DataFrame, config, to_mqtt=True):
        self.df = df
        self.info_col = config["info_col"]
        self.to_mqtt = to_mqtt
        self.config = config
        self.client = mqtt.Client(transport='websockets')
        # 設定登入帳號密碼
        self.client.username_pw_set(
            self.config['user'], self.config['password'])

        # 設定連線資訊(IP, Port, 連線時間)
        self.client.connect(
            self.config['mqtt_url'], self.config['mqtt_port'], 60)

    def time_domain(self, cols: list, window=10):
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

    def fft(self, cols: list, N=None, T=None):
        if N is None:
            N = len(self.df)
        print(N)
        if T is None:
            T = 60.0 / N
        fft_series = self.df.iloc[0][self.info_col]
        for idx, val in enumerate(cols):
            signal = self.df[val].to_numpy()
            fft_x = fftshift(fft(signal))
            fft_x_list = abs(fft_x).round(3).tolist()
            fft_series[val] = fft_x_list
            xf = fftshift(fftfreq(N, T))
            fft_series[val + '_ticks'] = xf.tolist()
        self.__send_mqtt(fft_series, 'vibration/fft')
        return fftfreq, fft_series

    def wavelet(self, cols: list, scales=np.rangea(1, 256), wavelet='morl'):
        cwt_series = self.df.iloc[0][self.info_col]
        for key, val in enumerate(cols):
            signal = self.df[val].to_numpy()
            coeffs, freqs = cwt(signal, scales, wavelet=wavelet)
            cwt_series['coeffs_'+val] = coeffs
            cwt_series['freqs_'+val] = freqs
        self.__send_mqtt(cwt_series, 'vibration/dwt')
        return cwt_series

    def envelope(self, cols: list, freq=4096):
        envelope_series = self.df.iloc[0][self.info_col]
        for key, val in enumerate(cols):
            signal = self.df[val].to_numpy()
            m_x = fft(signal)
            fft_x = np.abs(m_x)[:int(freq / 2)]
            fft_x_tranf = hilbert(fft_x)
            amplitude_envelope = np.sqrt(
                np.array(fft_x) ** 2 + fft_x_tranf ** 2)
            envelope_series[val] = np.abs(amplitude_envelope)
        self.__send_mqtt(envelope_series.transpose(), 'vibration/envelope')
        return envelope_series

    def stft(self, cols: list, fs=1e4, nperseg=1000):
        print(self.df.iloc[0])
        stft_series = self.df.iloc[0][self.info_col]
        cm = plt.cm.get_cmap('rainbow')
        for idx, val in enumerate(cols):
            signal = self.df[val].to_numpy()
            f, t, Zxx = stft(signal, fs, nperseg=nperseg)
            stft_series[val + '_' + 't'] = t
            stft_series[val + '_' + 'f'] = f
            stft_series[val + '_' + 'Zxx'] = np.abs(Zxx)
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

    def __to_mongo()
