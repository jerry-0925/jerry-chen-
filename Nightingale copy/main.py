import _pickle as cPickle

import numpy
import numpy as np
all_data = []
x = cPickle.load(open('data_preprocessed_python/s32.dat', 'rb'), encoding='iso-8859-1')
all_label = x['labels']
for i in x['data']:
    all_data.extend(i)
numpy.savetxt('subject32_data.csv', all_data, delimiter=',')
numpy.savetxt('subject32_label.csv',all_label, delimiter=",")



