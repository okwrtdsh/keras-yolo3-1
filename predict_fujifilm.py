#! /usr/bin/env python

import os
import argparse
import json
import cv2
from utils.utils import get_yolo_boxes, makedirs
from utils.bbox import draw_boxes
from keras.models import load_model
from tqdm import tqdm
import numpy as np


def write_boxes(file, filename, image, boxes, labels, obj_thresh, quiet=True):
    for box in boxes:
        label_str = ''
        label = -1

        for i in range(len(labels)):
            if box.classes[i] > obj_thresh:
                label_str = labels[i]
                label = i
            if not quiet: print(label_str)

        if label >= 0:
            file.write("""<annotation>
    <filename>{}</filename>
    <object>
        <name>{}</name>
        <bndbox>
            <xmin>{}</xmin>
            <ymin>{}</ymin>
            <xmax>{}</xmax>
            <ymax>{}</ymax>
        </bndbox>
    </object>
</annotation>
""".format(filename, label_str, max(box.xmin, 0), max(box.ymin, 0), min(box.xmax, 600), min(box.ymax, 600)))


def _main_(args):
    config_path  = args.conf
    input_path   = args.input
    output_path  = args.output

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    makedirs(output_path)

    ###############################
    #   Set some parameter
    ###############################       
    net_h, net_w = 416, 416 # a multiple of 32, the smaller the faster
    obj_thresh, nms_thresh = 0.5, 0.45

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
    with open(os.path.join(output_path, "ans.xml"), "w") as file:
        file.write("<annotations>\n")
        for image_path in image_paths:
            image = cv2.imread(image_path)
            print(image_path)

            # predict the bounding boxes
            boxes = get_yolo_boxes(infer_model, [image], net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)[0]

            # draw bounding boxes on the image using labels
            draw_boxes(image, boxes, config['model']['labels'], obj_thresh)

            filename = os.path.basename(image_path)
            write_boxes(file, filename, image, boxes, config['model']['labels'], obj_thresh)

            # write the image with bounding boxes to file
            cv2.imwrite(output_path + image_path.split('/')[-1], np.uint8(image))
        file.write("</annotations>\n")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
    argparser.add_argument('-c', '--conf', help='path to configuration file')
    argparser.add_argument('-i', '--input', help='path to an image, a directory of images, a video, or webcam')    
    argparser.add_argument('-o', '--output', default='output/', help='path to output directory')   
    
    args = argparser.parse_args()
    _main_(args)
