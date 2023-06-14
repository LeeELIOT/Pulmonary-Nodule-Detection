import xml.etree.ElementTree as ET
import pickle

from tqdm import tqdm
import os
from os import listdir, getcwd
from os.path import join

sets = [('train'), ('val'), ('test')]

classes = ["nodule"]


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)




for image_set in sets:
    if not os.path.exists('E:/graduateDesign/LUNA16/VOCdevkit/VOC2007/labels/' ):
        os.makedirs('E:/graduateDesign/LUNA16/VOCdevkit/VOC2007/labels/' )
    image_ids = open('E:/graduateDesign/LUNA16/VOCdevkit/VOC2007/ImageSets/Main/%s.txt' % (image_set)).read().strip().split()
    list_file = open('data/%s.txt' % (image_set), 'w')
    for image_id in tqdm(image_ids):
        print(image_id)
        list_file.write('E:/graduateDesign/LUNA16/VOCdevkit/VOC2007/JPEGImages/%s.png\n' % (image_id))
        in_file = open('E:/graduateDesign/LUNA16/VOCdevkit/VOC2007/Annotations/%s.xml' % (image_id), encoding='utf-8')
        out_file = open('E:/graduateDesign/LUNA16/VOCdevkit/VOC2007/labels/%s.txt' % (image_id), 'w', encoding='utf-8')
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        for obj in root.iter('object'):
            difficult = 0
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            bb = convert((w, h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

    list_file.close()

os.system("cat 2007_train.txt 2007_val.txt 2012_train.txt 2012_val.txt > train.txt")
os.system("cat 2007_train.txt 2007_val.txt 2007_test.txt 2012_train.txt 2012_val.txt > train.all.txt")
