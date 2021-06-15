import pandas as pd
import os
import json
from argparse import ArgumentParser
from transformer.vibration_transform import VibrationTransformation
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-p', type=str, required=True,
                        help="The path of json file")
    with open(r'config.json', 'r') as f:
        config = json.load(f)
    print(config)
    args = parser.parse_args()

    # df = pd.read_json(os.path.join(config["dir_path"], args.p.strip()))
    df = pd.read_json(args.p.strip())
    vt = VibrationTransformation(df, config=config)
    vt.stft(cols=config['col'])
    vt.fft(cols=config['col'])
    vt.envelope(cols=config['col'])
    vt.wavelet(cols=config['col'])
    vt.time_domain(cols=config['col'])
