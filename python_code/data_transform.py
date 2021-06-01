import pandas as pd
import os
import json
from argparse import ArgumentParser
from transformer.vibration_transform import VibrationTransformation




# cols=['x','y','z']
# info_col=['coil_id', 'schedule_id', 'routing_seq']
# dir_path =r'C:\Users\Edward\PycharmProjects\wavelet_POC\data\2021_01_05'
# data=load(os.path.join(dir_path, 'Crm1_vibration_DS_2021_01_05_11_00_00_2021_01_05_11_59_59.pkl'))
# sig_list= split_pass(data, len(data))
# print("splitting pass completed.")
# # for df in sig_list:
# vt = VibrationTransformation(sig_list[0], config=config)
# # vt.stft_and_fig(cols=cols, file_path=os.path.join(dir_path, 'stft'))
# # vt.fft_and_fig(cols=cols, file_path=os.path.join(dir_path, 'fft'))
# # vt.envelope_and_fig(cols=cols, file_path=os.path.join(dir_path, 'envelope'))
# # vt.wavelet_and_fig(cols=cols, file_path=os.path.join(dir_path, 'wavelet'))
# vt.time_domain(cols=cols, file_path=os.path.join(dir_path, 'time_domain'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-p', type=str, required=True, help="The path of json file")
    with open(r'D:\III\python_code\config.json', 'r') as f:
        config = json.load(f)
    print(config)
    args = parser.parse_args()

    df = pd.read_json(os.path.join(config["dir_path"], args.p))
    vt = VibrationTransformation(df, config=config)
    vt.stft_and_fig(cols=config['col'], file_path=os.path.join(config['dir_path'], 'stft'))
    vt.fft_and_fig(cols=config['col'], file_path=os.path.join(config['dir_path'], 'fft'))
    vt.envelope_and_fig(cols=config['col'], file_path=os.path.join(config['dir_path'], 'envelope'))
    # vt.wavelet_and_fig(cols=config['col'], file_path=os.path.join(config['dir_path'], 'wavelet'))
    vt.time_domain(cols=config['col'], file_path=os.path.join(config['dir_path'], 'time_domain'))