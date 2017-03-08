import load_data as data
import selective_search as select
import Image
import time
import numpy as np

positive_sample = []
negetive_sample = []

tag = [0 for i in range(209222)]

start_time = time.clock()
i = 0
while i<50000:
    this_i = np.random.randint(0,209222)
    if 1 == tag[this_i]:
        continue
    else:
        tag[this_i]=1
        img = Image.open(data.train_data[this_i])
        candidata = select.my_selective_search(img)
        #print candidata.__len__()
        target = data.img_bbox[data.train_data[this_i]]
        p_temp = str(this_i)+' '
        n_temp = str(this_i)+' '
        for c in candidata:
            IOU = select.get_IOU(target, c)
            if IOU >0.6:
                p_temp += ','.join(str(j)for j in c)+' '
            if IOU <0.3:
                n_temp += ','.join(str(j) for j in c)+' '
        positive_sample.append(p_temp)
        negetive_sample.append(n_temp)
        print i, this_i
        i+=1

print positive_sample
np.save('./detect_sample/positive', np.array(positive_sample))
np.save('./detect_sample/negetive', np.array(negetive_sample))
end_time = time.clock()
print 'generate_time:', end_time-start_time

