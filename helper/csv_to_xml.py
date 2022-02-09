import pandas as pd
import numpy as np
from lxml import etree
import xmlAnnotation.etree.cElementTree as ET


fields = ['NAME_ID', 'XMIN', 'YMIN', 'W', 'H', 'XMAX', 'YMAX']
df = pd.read_csv('loose_bb_test.csv', usecols=fields)


def nameChange(x):
    x = x.replace("/", "-")
    return x


df['NAME_ID'] = df['NAME_ID'].apply(nameChange)

for i in range(0, 2):
    height = df['H'].iloc[i]
    width = df['W'].iloc[i]
    depth = 3

    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = 'images'
    ET.SubElement(annotation, 'filename').text = str(df['NAME_ID'].iloc[i])
    ET.SubElement(annotation, 'segmented').text = '0'
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = str(depth)
    ob = ET.SubElement(annotation, 'object')
    ET.SubElement(ob, 'name').text = 'face'
    ET.SubElement(ob, 'pose').text = 'Unspecified'
    ET.SubElement(ob, 'truncated').text = '0'
    ET.SubElement(ob, 'difficult').text = '0'
    bbox = ET.SubElement(ob, 'bndbox')
    ET.SubElement(bbox, 'xmin').text = str(df['XMIN'].iloc[i])
    ET.SubElement(bbox, 'ymin').text = str(df['YMIN'].iloc[i])
    ET.SubElement(bbox, 'xmax').text = str(df['XMAX'].iloc[i])
    ET.SubElement(bbox, 'ymax').text = str(df['YMAX'].iloc[i])

    fileName = str(df['NAME_ID'].iloc[i])
    tree = ET.ElementTree(annotation)
    tree.write(fileName + ".xml", encoding='utf8')