import load_data as data
import selective_search as select
import Image
import time
import numpy as np

positive_sample = [[]]
negetive_sample = [[]]

start_time = time.clock()
n = 0
for i in range(209222):
    img = Image.open(data.train_data[i])
    candidata = select.my_selective_search(img)
    #print candidata.__len__()
    target = data.img_bbox[data.train_data[i]]
    print n
    p_temp = ''
    n_temp = ''
    for c in candidata:
        IOU = select.get_IOU(target, c)
        if IOU >0.6:
            p_temp += ','.join(str(j)for j in c)
        if IOU <0.3:
            n_temp += ','.join(str(j) for j in c)
    positive_sample.append(p_temp)
    negetive_sample.append(n_temp)
    n+=1

np.save('./detect_sample/positive', np.array(positive_sample))
np.save('./detect_sample/negetive', np.array(negetive_sample))
end_time = time.clock()
print 'generate_time:', end_time-start_time

