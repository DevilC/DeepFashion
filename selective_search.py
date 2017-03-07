import selectivesearch as select
from PIL import Image
from PIL import ImageDraw
import numpy as np
import math
import load_data as data

def my_selective_search(img):
    img_array = np.array(img)
    img_size = img_array.size/3    #the number of img pixel
    min_region_size = img_size/100.0
    min_component_size = int(math.sqrt(img_size)/10)

    #img_array = data.astronaut()
    img_lbl, regions = select.selective_search(img_array, scale = 300,sigma=0.9, min_size = min_component_size)
    #draw = ImageDraw.Draprint get_IOU([50, 50, 100 ,0], [0,0,100,100])w(img)
    #print regions.__len__()
    candidate = set()
    n = 0
    for i in regions:
        if i['rect'] in candidate:
            continue
        if i['size'] <min_region_size:
            continue
        x, y, w, h = i['rect']
        if w/h >2 or h/w >2:
            continue
        rect = (x, y, x+w, y+h)
        candidate.add(rect)
    return candidate
'''''
    draw = ImageDraw.Draw(img)
    draw.rectangle(data.img_bbox[data.train_data[57]], outline='red')
    img.show()
'''''

def get_IOU(img_a,img_b):
    #get left-up point and right-down point
    x1, y1, x2, y2 = img_a
    x11 = min(x1, x2)
    x12 = max(x1, x2)
    y11 = min(y1, y2)
    y12 = max(y1, y2)
    x1, y1, x2, y2 = img_b
    x21 = min(x1, x2)
    x22 = max(x1, x2)
    y21 = min(y1, y2)
    y22 = max(y1, y2)
    #count area of A and B
    area_a = (x12-x11)*(y12-y11)
    area_b = (x22-x21)*(y22-y21)
    #get cross area coordinate(x1, y1), (x2, y2)
    x1 = max(x11, x21)
    y1 = max(y11, y21)
    x2 = min(x12, x22)
    y2 = min(y12, y22)
    cross_area = 0
    if x2-x1>0 and y2-y1>0:
        cross_area = (x2-x1)*(y2-y1)
    IOU = float(cross_area)/(area_a + area_b-cross_area)
    return IOU