import os
import argparse
import datetime
import pickle
import xml.etree.ElementTree as ET
from difflib import get_close_matches, ndiff
from glob import glob
from functools import lru_cache

import numpy as np
from sklearn.cluster import DBSCAN


def read_annotation(ann_filename):
    annotation = {
        'bboxes': None,
        'label': None
    }
    tree = ET.parse(ann_filename)

    objects = []
    for elem in tree.iter():
        if 'object' in elem.tag:
            obj = {}
            for attr in list(elem):
                if 'score' in attr.tag:
                    obj['score'] = float(attr.text)
                if 'name' in attr.tag:
                    obj['char'] = attr.text
                if 'bndbox' in attr.tag:
                    for dim in list(attr):
                        if 'xmin' in dim.tag:
                            obj['xmin'] = int(round(float(dim.text)))
                        if 'ymin' in dim.tag:
                            obj['ymin'] = int(round(float(dim.text)))
                        if 'xmax' in dim.tag:
                            obj['xmax'] = int(round(float(dim.text)))
                        if 'ymax' in dim.tag:
                            obj['ymax'] = int(round(float(dim.text)))
            objects.append(obj)
        if 'label' in elem.tag:
            annotation['label'] = elem.text
    annotation['bboxes'] = objects
    return annotation


def fujifilm_score(y_true, y_pred):
    if y_true == y_pred:
        return 1
    t_list = y_true.split()
    p_list = y_pred.split()
    return 2 * len([line for line in ndiff(t_list, p_list) if line[0] == ' ']) / (len(t_list) + len(p_list))


def _main_(args):
    images_path = args.images
    annotations_path = args.annotations
    output_path = args.output
    is_train = args.stage == 'train'
    cutoff = args.cutoff
    min_score = args.min_score
    min_score_frame = args.min_score_frame
    padding = args.padding_frame
    weight_diff = args.weight_diff
    weight_width = args.weight_width
    cache_name = args.cache_name
    # n_prev_preds = args.n_prev_preds

    cache_path = os.path.join(annotations_path, cache_name)
    if os.path.isfile(cache_path):
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
    else:
        cache = []
        for path in sorted(glob(os.path.join(images_path, "*.jpg"))):
            path_xml = path.replace(images_path, annotations_path).replace('jpg', 'xml')
            obj = {
                'path': path,
                'filename': os.path.basename(path),
                'path_xml': path_xml,
                'exists': os.path.isfile(path_xml),
                'ann': None
            }
            if os.path.isfile(path_xml):
                obj['ann'] = read_annotation(path_xml)
            cache.append(obj)
        with open(cache_path, 'wb') as f:
            pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 富士フィルムで日付写し込み機能を搭載したのは1980年11月
    date = datetime.date(1980, 11, 1)
    strs = ['']
    while date != datetime.date(2019, 1, 1):
        if date < datetime.date(2010, 1, 1):
            strs.append(date.strftime("%y %m %d"))
            strs.append(date.strftime("%y %-m %-d"))
            strs.append(date.strftime("%d %m %y"))
            strs.append(date.strftime("%-d %-m %y"))
            strs.append(date.strftime("%m %d %y"))
            strs.append(date.strftime("%-m %-d %y"))
        if date > datetime.date(2000, 1, 1):
            strs.append(date.strftime("%Y %m %d"))
            strs.append(date.strftime("%Y %-m %-d"))
        date += datetime.timedelta(days=1)
    strs = sorted(set(strs))

    @lru_cache(maxsize=1024)
    def _get_close_matches(db_label, cutoff, n):
        return get_close_matches(db_label, strs, cutoff=cutoff, n=n)

    if is_train:
        y_true = []
        scores = []
    y_pred = []
    filenames = []
    # prev_preds = []
    for obj in cache:
        path = obj['path']
        exists = obj['exists']
        filenames.append(obj['filename'])
        if not exists:
            if is_train:
                y_true.append('')
            y_pred.append('')
        else:
            ann = obj['ann']
    # for path in sorted(glob(os.path.join(images_path, "*.jpg"))):
    #     filenames.append(os.path.basename(path))
    #     path_xml = path.replace(images_path, annotations_path).replace('jpg', 'xml')
    #     if not os.path.isfile(path_xml):
    #         if is_train:
    #             y_true.append('')
    #         y_pred.append('')
    #     else:
    #         ann = read_annotation(path_xml)
            if is_train:
                y_true.append(ann['label'])
            boxes = []
            chars = []
            frames = []
            for box in ann['bboxes']:
                if box['char'] != 'frame':
                    continue
                if 'score' in box and box['score']/100 < min_score_frame:
                    continue
                frames.append(box)
            for frame in frames:
                boxes.append([])
                chars.append([])
                for box in ann['bboxes']:
                    if box['char'] not in '0123456789':
                        continue
                    if 'score' in box and box['score']/100 < min_score:
                        continue
                    if frame['xmin'] - padding < (box['xmin'] + box['xmax'])/2 < frame['xmax'] + padding and\
                            frame['ymin'] - padding < (box['ymin'] + box['ymax']) / 2 < frame['ymax'] + padding:
                        boxes[-1].append([
                            box['xmin'],
                            box['ymin'],
                            box['xmax'],
                            box['ymax'],
                        ])
                        chars[-1].append(box['char'])

            if len(frames) == 0:
                boxes = []
                chars = []
            elif len(frames) == 1:
                boxes = boxes[0]
                chars = chars[0]
            elif len(frames) == 2:
                f1 = len(boxes[0])
                f2 = len(boxes[1])
                if f1 < 4 and f2 < 4:
                    boxes = []
                    chars = []
                elif f1 < 4 and f2 > 3:
                    boxes = boxes[1]
                    chars = chars[1]
                elif f2 < 4 and f1 > 3:
                    boxes = boxes[0]
                    chars = chars[0]
                else:
                    if frames[0]['score'] > frames[1]['score']:
                        boxes = boxes[0]
                        chars = chars[0]
                    else:
                        boxes = boxes[1]
                        chars = chars[1]
            else:
                index = None
                len_chars = []
                for i, (chrs, frame) in enumerate(zip(chars, frames)):
                    len_chars.append(len(chrs))
                    if frame['score'] > 99:
                        if '9' in chrs:
                            index = i
                if index is not None:
                    boxes = boxes[index]
                    chars = chars[index]
                elif all(l < 4 for l in len_chars):
                    boxes = []
                    chars = []
                else:
                    for i, l in enumerate(len_chars):
                        if l > 3:
                            boxes = boxes[i]
                            chars = chars[i]

            if len(boxes) < 4:
                if is_train:
                    y_true.append('')
                y_pred.append('')
            else:
                width = np.mean([b[2]-b[0] for b in boxes])
                diff = np.mean(np.diff([(b[2]+b[0])/2 for b in boxes]))
                db = DBSCAN(min_samples=1, eps=diff*weight_diff + width*weight_width)
                db.fit(boxes)
                prev = db.labels_[0]
                db_label = ''
                for i, k in enumerate(db.labels_):
                    if k != prev:
                        db_label += ' '
                    prev = k
                    db_label += chars[i]

                res = _get_close_matches(db_label, cutoff=cutoff, n=10)
                gpm_db_label = res[0] if len(res) > 0 else ''
                y_pred.append(gpm_db_label)
                # if len(prev_preds) >= n_prev_preds:
                #     if prev_preds[-1] in res:
                #         gpm_db_label = prev_preds[-1]
                #         prev_preds.append(gpm_db_label)
                #     else:
                #         gpm_db_label = res[0] if len(res) > 0 else ''
                #         prev_preds = [gpm_db_label]
                #     y_pred.append(gpm_db_label)
                # else:
                #     gpm_db_label = res[0] if len(res) > 0 else ''
                #     y_pred.append(gpm_db_label)
                #     if prev_preds and prev_preds[-1] == gpm_db_label:
                #         prev_preds.append(gpm_db_label)
                #     else:
                #         prev_preds = [gpm_db_label]

        if is_train:
            scores.append(fujifilm_score(y_true[-1], y_pred[-1]))
            print(path, ':\t', y_pred[-1], '\t', y_true[-1], '\t', scores[-1])
        else:
            print(path, ':\t', y_pred[-1])

    if is_train:
        print(np.mean(scores))
    else:
        with open(output_path, 'w') as f:
            for name, pred in zip(filenames, y_pred):
                f.write("%s,%s\n" % (name, pred))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-i', '--images', help='input dir')
    argparser.add_argument('-a', '--annotations', help='annotation dir')
    argparser.add_argument('-o', '--output', default='submit.csv', help='output file')
    argparser.add_argument('-s', '--stage', default='test', help='stage train or test')
    argparser.add_argument('-c', '--cutoff', type=float, default=0.8, help='cutoff')
    argparser.add_argument('-m', '--min_score', type=float, default=0.6, help='min_score')
    argparser.add_argument('-fm', '--min_score_frame', type=float, default=0.6, help='min_score')
    argparser.add_argument('-fp', '--padding_frame', type=int, default=10, help='padding_frame')
    argparser.add_argument('-wd', '--weight_diff', type=float, default=1.0, help='weight_diff')
    argparser.add_argument('-ww', '--weight_width', type=float, default=0.5, help='weight_width')
    # argparser.add_argument('-p', '--n_prev_preds', type=int, default=2, help='n_prev_preds')
    argparser.add_argument('-cache', '--cache_name', default='cache.pkl', help='cache_name')
    args = argparser.parse_args()
    _main_(args)
