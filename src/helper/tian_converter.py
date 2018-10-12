import os, sys
import struct
from tensorflow.core.example import example_pb2

base_dir = '/home/tshi/summarization/newsroom/newsroom_data/nltk_data/sum_data/'
root_dir = '/home/yaserkl/data/tian/'

for filetype in ['train','val','test']:
    f = open('{}/{}.txt'.format(base_dir, filetype))
    pwriter = open('{}/{}.bin'.format(root_dir,filetype), 'wb')
    for line in f:
        abstract, article = line.strip().split('<sec>')
        tf_example = example_pb2.Example()
        tf_example.features.feature['article'].bytes_list.value.extend([article.encode()])
        tf_example.features.feature['abstract'].bytes_list.value.extend([abstract.encode()])
        tf_example_str = tf_example.SerializeToString()
        str_len = len(tf_example_str)
        pwriter.write(struct.pack('q', str_len))
        pwriter.write(struct.pack('%ds' % str_len, tf_example_str))
    pwriter.close()
    f.close()