import os
import json
from glob import glob

import numpy as np
from PIL import Image


def _main_(args):
    input_path = args.input_path
    annotation_path = args.annotation_path
    fujifilm_ano_path = args.fujifilm_ano_path
    ano_only = args.ano_only
    with_frame = args.with_frame
    os.makedirs(annotation_path, mode=0o777, exist_ok=True)
    print(ano_only)

    for path in sorted(glob(os.path.join(input_path, "*.jpg"))):
        print(path)
        path_json = path.replace(input_path, fujifilm_ano_path).replace('jpg', 'json')
        path_xml = path.replace(input_path, annotation_path).replace('jpg', 'xml')

        if os.path.exists(path_json):
            with open(path_json, "r") as f:
                anns = json.load(f)
        else:
            if not ano_only:
                anns = None
            else:
                continue

        img = np.array(Image.open(path))
        basename = os.path.basename(path)
        dirname = os.path.basename(os.path.dirname(path))

        with open(path_xml, "w") as f:
            # basic info
            f.write("""<annotation>
    <folder>{}</folder>
    <filename>{}</filename>
    <path>{}</path>""".format(dirname, basename, os.path.join(dirname, basename)))
            # image info
            f.write("""
    <size>
        <width>{1}</width>
        <height>{0}</height>
        <depth>{2}</depth>
    </size>""".format(*img.shape))

            if anns is not None:
                # annotation info
                f.write("""
    <label>{}</label>""".format(anns['label']))
                for box in anns['bboxes']:
                    # box info
                    f.write("""
    <object>
        <name>{char}</name>
        <bndbox>
            <xmin>{xmin}</xmin>
            <ymin>{ymin}</ymin>
            <xmax>{xmax}</xmax>
            <ymax>{ymax}</ymax>
        </bndbox>
    </object>""".format(**box))

                if with_frame:
                    xmins = []
                    ymins = []
                    xmaxs = []
                    ymaxs = []
                    for box in anns['bboxes']:
                        xmins.append(box['xmin'])
                        ymins.append(box['ymin'])
                        xmaxs.append(box['xmax'])
                        ymaxs.append(box['ymax'])
                    box = {
                        'char': 'frame',
                        'xmin': min(xmins),
                        'ymin': min(ymins),
                        'xmax': max(xmaxs),
                        'ymax': max(ymaxs),
                    }
                    f.write("""
    <object>
        <name>{char}</name>
        <bndbox>
            <xmin>{xmin}</xmin>
            <ymin>{ymin}</ymin>
            <xmax>{xmax}</xmax>
            <ymax>{ymax}</ymax>
        </bndbox>
    </object>""".format(**box))

            f.write("\n</annotation>")


if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        '-i',
        '--input_path',
        default='../input/train_images',
        help='input path')
    argparser.add_argument(
        '-a',
        '--annotation_path',
        default='../input/train_annotations',
        help='annotation path')
    argparser.add_argument(
        '-f',
        '--fujifilm_ano_path',
        default='../input/train_anns',
        help='fujifilm annotation path')
    argparser.add_argument(
        '--ano_only',
        default=False,
        action='store_true',
        help='annotation_only')
    argparser.add_argument(
        '--with_frame',
        default=False,
        action='store_true',
        help='with frame')

    args = argparser.parse_args()
    _main_(args)
