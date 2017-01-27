from PIL import Image
import numpy
import time

'''''
--load_data--
get data from file
-----------------------------
read_img(path)---input the path of img ,select the area of box ,resize to 100*100 and return an array of img
-----------------------------
read_bbox()---read list_bbox.txt and save data in img_bbox (type:dict)
-----------------------------
get_attr()---read list_attr_img.txt and save data in img_attr (type:dict)
-----------------------------
read_Eval()---read list_eval_partition.txt and save train_data ,val_data and test_data(type: list)
'''''
data_path = r'/home/lu/MyDocument/Category and Attribute Prediction Benchmark/Img/'
bbox_path = r'/home/lu/MyDocument/Category and Attribute Prediction Benchmark/Anno/list_bbox.txt'
attr_path = r'/home/lu/MyDocument/Category and Attribute Prediction Benchmark/Anno/list_attr_img.txt'
Eval_path = r'/home/lu/MyDocument/Category and Attribute Prediction Benchmark/Eval/list_eval_partition.txt'
img_attr = {}
img_bbox = {}
train_data = []
test_data = []
val_data = []

#read image,cut the clothes area of the image,resize to 100*100 and return an array
def read_img(path):
    try:
        img = Image.open(path)
        img = img.crop(img_bbox[path])
        img = img.resize((100,100))
        img.show()
        i_array = numpy.array(img)
        return i_array
    except Exception:
        print "read_img Error!"
'''''
r = read_img(r'./img_00000002.jpg')[:,:,2]
a = Image.fromarray(r)
a.save(r'./img_00000002_2.png')
'''''

#read list_bbox,save as a dict
def read_bbox():
    try:
        bbox_file = open(bbox_path, 'r')
    except Exception:
        print "list_bbox.txt IOError!"
    bbox_data = bbox_file.readlines()
    for d, i in zip(bbox_data[2:],range(bbox_data.__len__()-2)):
        s = d.split()
        p = data_path + s[0]
        bbox_loca = []
        for a in s[1:]:
            bbox_loca.append(float(a))
        img_bbox[p] = bbox_loca
    bbox_file.close()
    print 'list_bbox.txt ready!'

def get_attr():
    attr_file = open(attr_path, 'r')
    for i in range(289224):
        if i>2:
            l = attr_file.readline()
            s = l.split()
            p = data_path + s[0]
            img_attr[p] = ' '.join(s[1:])
    print 'list_attr_img.txt ready!'


def read_Eval():
    eval_file = open(Eval_path , 'r')
    lines = eval_file.readlines()
    for i in lines[2:]:
        s = i.split()
        type = s[1]
        if type == 'train':
            train_data.append(data_path+s[0])
        elif type == 'test':
            test_data.append(data_path+s[0])
        else:
            val_data.append(data_path+s[0])
    print 'list_eval_partition.txt ready!!'


print '--*-------------------------------*--\n---------------data_load---------------\n--*-------------------------------*--'
start_time = time.clock()
read_Eval()
eval_time = time.clock()
read_bbox()
bbox_time = time.clock()
get_attr()
attr_time = time.clock()
end_time = time.clock()
print 'train_data size:', train_data.__len__()
print 'test_data size:', test_data.__len__()
print 'val_data size:', val_data.__len__()
print 'eval time:', eval_time-start_time, 'sec'
print 'bbox time:', bbox_time-eval_time, 'sec'
print 'attr time:', attr_time-bbox_time, 'sec'
print 'data_load run time:', end_time-start_time, 'sec'
print '--*-------------------------------*--\n-------------data_load finish-------------\n--*-------------------------------*--'
