import pandas as pd
import os
import json
from argparse import ArgumentParser
from vibration_transform import VibrationTransformation
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-p', type=str, required=True,
                        help="The path of json file")
    parser.add_argument('-c', type=str, default='config.json', help='config file path. Default is config.json in the same dir.')
    args = parser.parse_args()

    # 讀取設定檔案 
    with open(args.c, 'r') as f:
        config = json.load(f)
    print(config)
    
    df = pd.read_json(os.path.join(config["dir_path"], args.p.strip()))
    # df = pd.read_json(args.p.strip())
    vt = VibrationTransformation(df, config=config)
    vt.stft(cols=config['col'])
    vt.time_domain(cols=config['col'])
    vt.fft(cols=config['col'])
    vt.envelope(cols=config['col'])
    vt.wavelet(cols=config['col'])
    # vt.time_domain(cols=config['col'])
