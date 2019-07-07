import os
from glob import glob

import numpy as np
import cv2


def _main_(args):
    input_dir_path = args.input_dir
    output_dir_path = args.output_dir
    os.makedirs(output_dir_path, mode=0o777, exist_ok=True)
    gamma = args.gamma
    print(output_dir_path)
    print(gamma)

    for path in sorted(glob(os.path.join(input_dir_path, "*.jpg"))):
        print(path)
        path_output = path.replace(input_dir_path, output_dir_path)
        img = cv2.imread(path)
        img_gamma = cv2.LUT(img, (255 * (np.arange(256) / 255) ** gamma).reshape(256, 1))
        cv2.imwrite(path_output, np.uint8(img_gamma))


if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        '-i',
        '--input_dir',
        help='input dir path')
    argparser.add_argument(
        '-o',
        '--output_dir',
        help='output dir path')
    argparser.add_argument(
        '-g',
        '--gamma',
        default=4,
        type=float,
        help='gamma value')

    args = argparser.parse_args()
    _main_(args)
