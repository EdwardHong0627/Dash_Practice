import pandas as pd
import os
import sys
sys.path.insert(0,'/py_script')
import json
from argparse import ArgumentParser
from transformer.vibration_transform import VibrationTransformation







if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-p', type=str, required=True, help="The path of json file")
    with open(r'config.json', 'r') as f:
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