import os
import glob
import xml.etree.ElementTree as ET
import shutil
import random
from collections import defaultdict
import numpy as np

# RDD2022 target classes (often D00, D10, D20, D40)
CLASSES = ['D00', 'D10', 'D20', 'D40']

def convert_to_yolo_bbox(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    
    bboxes = []
    classes = []
    for obj in root.findall('object'):
        cls = obj.find('name').text
        if cls not in CLASSES:
            continue
        cls_id = CLASSES.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
             float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        yolo_box = convert_to_yolo_bbox((w, h), b)
        bboxes.append(yolo_box)
        classes.append(cls_id)
    return classes, bboxes

# Script continues...
