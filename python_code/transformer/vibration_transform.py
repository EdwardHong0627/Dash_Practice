from logging import log
import re
import pandas as pd
from pywt import cwt
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import hilbert
import numpy as np
import paho.mqtt.client as mqtt
from scipy.signal import stft
import json
from pymongo import MongoClient
from loguru import logger


class VibrationTransformation(object):
    """
    震動訊號頻域時域轉換
    Input:
        df: dataframe
        config: config dict
        to_mqtt: whether to send to mqtt
    """
    def __init__(self, df: pd.DataFrame, config:dict, to_mqtt:bool=True):
        self.df = df
        self.info_col = config["info_col"]
        self.to_mqtt = to_mqtt
        self.config = config
        self.mqtt_client = mqtt.Client(transport='websockets')
        # 設定登入帳號密碼
        self.mqtt_client.username_pw_set(
            self.config['mqtt_user'], self.config['mqtt_password'])

        # 設定連線資訊(IP, Port, 連線時間)
        self.mqtt_client.connect(
            self.config['mqtt_url'], self.config['mqtt_port'], 60)
        self.mongo_db = MongoConnector().db_connect()



    def time_domain(self, cols: list, window=60):
        """
        Time domain features calculate mean, std, min, max ,skew, 
            kurt, rms, and pip with window default 60 seconds.
        Input: 
            cols: column list which should be transformed in df
            window: time interval used in aggregation.
        """
        try:
            info_data = self.df[self.info_col].copy()
            raw_data = self.df[cols].copy()
            statis_data = raw_data.rolling(window=window).agg(['mean', 'std', 'min', 'max',
                                                               'skew', 'kurt', self.rms, self.ptp])
            statis_data.columns = [i[1] + '_' + i[0]
                                   for i in statis_data.columns]
            statis_data = statis_data.dropna()
            all_data = pd.concat([info_data, statis_data], axis=1)
            all_data = all_data.dropna()
            all_data = all_data.reset_index()
            all_data=all_data.round(3)
            time_domain_json = all_data.to_dict('records')
            print(time_domain_json)
            self.__send_mqtt(time_domain_json, 'vibration/time')
            self.mongo_db['time_domain'].insert_many(time_domain_json)
            return all_data
        except Exception as err:
            logger.error("An error occurs when transforming {}:{}".format(
                'time_domain', err))

    def fft(self, cols: list, N=None, T=None):
        """
        Fast fourier transform of input columns
        Input:
            cols: column list
            N: window size
            T: sample size default with len of dataframe, if data volume is too much can turn it lower to reduce output size.
            Can refer to: https://docs.scipy.org/doc/scipy/reference/reference/generated/scipy.fft.fftfreq.html#scipy.fft.fftfreq
        Outout:
            Save freqlist and value in the corresponding freq with timestamp into MongoDB (2 lists and one timestamp)
        """
        try:
            if N is None:
                N = 100
            if T is None:
                T = 1/400.0
            fft_series = self.df.iloc[0][self.info_col]
            for idx, val in enumerate(cols):
                signal = self.df[val].to_numpy()
                fft_x = fftshift(fft(signal))
                fft_x_list = abs(fft_x).round(3).tolist()
                fft_series[val] = fft_x_list
                xf = fftshift(fftfreq(N, T))
                fft_series[val + '_ticks'] = xf.round(3).tolist()
            fft_json = self.__results_to_dict(fft_series)
            self.__send_mqtt(fft_json, 'vibration/fft')
            self.mongo_db['fft'].insert_one(fft_json)
            return fftfreq, fft_series
        except Exception as err:
            logger.error(
                "An error occurs when transforming {}:{}".format('fft', err))

    def wavelet(self, cols: list, scales=np.arange(1, 10), wavelet='morl'):
        """
        Wavelet transform of input columns
        Input:
            cols: column list
            scales: scaler of mother wavelet (以基準一倍當作模擬)
            wavelet: The type of mother wavelet
            Can refer to: https://pywavelets.readthedocs.io/en/latest/ref/cwt.html
        Outout:
            Save scaler list, time interval and value in the corresponding freq with timestamp into MongoDB (2 lists, a 2D matrix and one timestamp)
        """
        try:
            cwt_series = self.df.iloc[0][self.info_col]
            for key, val in enumerate(cols):
                signal = self.df[val].to_numpy()
                coeffs, freqs = cwt(signal, scales, wavelet=wavelet)
                cwt_series['coeffs_'+val] = coeffs.round(3).tolist()
                cwt_series['freqs_'+val] = freqs.round(3).tolist()
            wavelet_json = self.__results_to_dict(cwt_series)
            self.__send_mqtt(wavelet_json, 'vibration/dwt')
            self.mongo_db['wavelet'].insert(wavelet_json)
            return cwt_series
        except Exception as err:
            logger.error(
                "An error occurs when transforming {}:{}".format('wavelet', err))

    def envelope(self, cols: list, freq=4096):
        """
        Energy of Fast fourier transform of input columns
        Input:
            cols: column list
            freq: sample fraq
        Outout:
            a list containing an energy value of wave every freq.
            (把它當成FFT的逼近取線就對惹)
        """
        try:
            envelope_series = self.df.iloc[0][self.info_col]
            for key, val in enumerate(cols):
                signal = self.df[val].to_numpy()
                m_x = fft(signal)
                fft_x = np.abs(m_x)[:int(freq / 2)]
                fft_x_tranf = hilbert(fft_x)
                amplitude_envelope = np.sqrt(
                    np.array(fft_x) ** 2 + fft_x_tranf ** 2)
                envelope_series[val] = np.abs(
                    amplitude_envelope).round(3).tolist()
            envelope_json = self.__results_to_dict(envelope_series)
            self.__send_mqtt(envelope_json, 'vibration/envelope')
            self.mongo_db['envelope'].insert(envelope_json)
            return envelope_series
        except Exception as err:
            logger.error(
                "An error occurs when transforming {}:{}".format('envelope', err))

    def stft(self, cols: list, **kwargs):
        try:
            print(self.df.iloc[0])
            stft_series = self.df.iloc[0][self.info_col]
            cm = plt.cm.get_cmap('rainbow')
            for idx, val in enumerate(cols):
                signal = self.df[val].to_numpy()
                f, t, Zxx = stft(signal, **kwargs)
                stft_series[val + '_' + 't'] = t.tolist()
                stft_series[val + '_' + 'f'] = f.tolist()
                stft_series[val + '_' + 'Zxx'] = np.abs(Zxx).round(3).tolist()
            stft_json = self.__results_to_dict(stft_series)
            self.mongo_db['stft'].insert(stft_json)
            self.__send_mqtt(stft_json, 'vibration/stft')
            return stft_series
        except Exception as err:
            logger.error(
                "An error occurs when transforming {}:{}".format('stft', err))

    def __send_mqtt(self, x, topic):
        def date_conv(o):
            if isinstance(o, datetime):
                return o.__str__()
        if self.to_mqtt:
            try:
                self.mqtt_client.publish(topic, json.dumps(x, default=date_conv))
            except Exception as err:
                logger.error(
                    "An error occurs when transmitting {} messages:{}".format('mqtt', err))

    def __results_to_dict(self, result):
        try:
            x = result.to_dict()
            return x
        except Exception as err:
            logger.error(
                "Occurs an error while transforming df to json:{}".format(err))

    def rms(self, x):
        return np.sqrt(np.mean(np.power(x.values, 2)))

    def ptp(self, x):
        return np.ptp(x.values)


class MongoConnector(object):
    def __init__(self, uri: str = 'mongodb://iii:mis-12345@192.168.1.219:27017/', user: str = 'iii', passwd: str = 'mis-12345', db: str = 'vibration'):
        self.uri = uri
        self.user = user
        self.passwd = passwd
        self.db = db
        self.connection = self.db_connect()

    def db_connect(self):
        client = MongoClient(self.uri)
        db = client[self.db]
        return db
