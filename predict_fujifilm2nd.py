#! /usr/bin/env python

import os
import argparse
import json
import cv2
from utils.utils import get_yolo_boxes, makedirs
from utils.bbox import draw_boxes
from keras.models import load_model
import numpy as np
from PIL import Image


def write_annos(path, path_xml, boxes, labels, obj_thresh):
    img = np.array(Image.open(path))
    basename = os.path.basename(path)
    dirname = os.path.basename(os.path.dirname(path))

    cnt_box = 0
    for box in boxes:
        val_max = -1
        for i in range(len(labels)):
            if box.classes[i] > obj_thresh and val_max < box.classes[i]:
                val_max = box.classes[i]
        if val_max > 0:
            cnt_box += 1

    if cnt_box > 20 or cnt_box < 1:
        return

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

        for box in sorted(boxes, key=lambda box: (box.xmin, box.ymin)):
            label_str = ''
            val_max = -1
            for i in range(len(labels)):
                if box.classes[i] > obj_thresh and val_max < box.classes[i]:
                    label_str = labels[i]
                    val_max = box.classes[i]
            if label_str:
                # box info
                f.write("""
    <object>
        <name>{char}</name>
        <score>{score}</score>
        <bndbox>
            <xmin>{xmin}</xmin>
            <ymin>{ymin}</ymin>
            <xmax>{xmax}</xmax>
            <ymax>{ymax}</ymax>
        </bndbox>
    </object>""".format(**{
                    'char': label_str,
                    'xmin': max(box.xmin, 0),
                    'ymin': max(box.ymin, 0),
                    'xmax': min(box.xmax, 600),
                    'ymax': min(box.ymax, 600),
                    'score': str(round(box.get_score()*100, 2))
                }))

        f.write("\n</annotation>")


def _main_(args):
    config_path = args.conf
    input_path = args.input
    output_path = args.output

    with open(config_path) as config_buffer:
        config = json.load(config_buffer)

    makedirs(output_path)

    ###############################
    #   Set some parameter
    ###############################
    net_h, net_w = 416, 416  # a multiple of 32, the smaller the faster
    obj_thresh, nms_thresh = float(config['test']['obj_thresh']), float(config['test']['nms_thresh'])

    ###############################
    #   Load the model
    ###############################
    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    infer_model = load_model(config['train']['saved_weights_name'])

    ###############################
    #   Predict bounding boxes
    ###############################
    image_paths = []

    if os.path.isdir(input_path):
        for inp_file in os.listdir(input_path):
            image_paths += [input_path + inp_file]
    else:
        image_paths += [input_path]

    image_paths = sorted([inp_file for inp_file in image_paths if (inp_file[-4:] in ['.jpg', '.png', 'JPEG'])])

    # the main loop
    for image_path in image_paths:
        image = cv2.imread(image_path)
        print(image_path)

        # predict the bounding boxes
        boxes = get_yolo_boxes(
            infer_model, [image], net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)[0]

        # draw bounding boxes on the image using labels
        if not args.ano_only:
            draw_boxes(image, boxes, config['model']['labels'], obj_thresh)
        path_xml = image_path.replace(input_path, output_path).replace('jpg', 'xml')
        write_annos(image_path, path_xml, boxes, config['model']['labels'], obj_thresh)

        # filename = os.path.basename(image_path)

        # write the image with bounding boxes to file
        if not args.ano_only:
            cv2.imwrite(output_path + image_path.split('/')[-1], np.uint8(image))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
    argparser.add_argument('-c', '--conf', help='path to configuration file')
    argparser.add_argument('-i', '--input', help='path to an image, a directory of images, a video, or webcam')
    argparser.add_argument('-o', '--output', default='output/', help='path to output directory')
    argparser.add_argument('--ano_only', default=False, action='store_true', help='annotation_only')

    args = argparser.parse_args()
    _main_(args)
