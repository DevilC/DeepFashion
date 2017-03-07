import load_data as data
import selective_search as select
import Image
import time
import numpy as np

positive_sample = []
negetive_sample = []

start_time = time.clock()
n = 0
for i in range(209222):
    positive_sample.append([])
    negetive_sample.append([])
    img = Image.open(data.train_data[i])
    candidata = select.my_selective_search(img)
    #print candidata.__len__()
    target = data.img_bbox[data.train_data[i]]
    print n
    for c in candidata:
        IOU = select.get_IOU(target, c)
        if IOU >0.6:
            positive_sample[i].append(c)
        if IOU <0.3:
            negetive_sample[i].append(c)
    n+=1

np.save('./detect_sample/positive', np.array(positive_sample))
np.save('./detect_sample/positive', np.array(negetive_sample))
end_time = time.clock()
print 'generate_time:', end_time-start_time

