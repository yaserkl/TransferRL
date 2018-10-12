# python src/elmo_embedding.py ~/data/tian/vocab-50k ~/data/tian/vocab_elmo-50k.emb 8192

import tensorflow_hub as hub
import sys
import tensorflow as tf
import numpy as np

exp_name = sys.argv[1]
vocab_file = sys.argv[2]
out_file = sys.argv[3]
embedding_file = sys.argv[4]
emb_dim = int(sys.argv[5])

def _run_elmo():
    ### embedding_file = "https://tfhub.dev/google/elmo/2"
    elmo = hub.Module(embedding_file, trainable=True)

    words = [word.strip().split()[0] for word in open(vocab_file).readlines()]
    gpu_options = tf.GPUOptions(allow_growth=True)
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)

    fw = open(out_file, 'w')
    fw.write('{} {}\n'.format(len(words), emb_dim))
    batch_size = 4096
    with tf.Session(config=session_config) as sess:
        i = 0
        while i < len(words):
            print('batch {}/{}...'.format(i/batch_size, len(words)/batch_size))
            _words = [words[i:(i+batch_size)]]
            embeddings = elmo(
                inputs={
                    "tokens": _words,
                    "sequence_len": [1]*len(_words)
                },
                signature="tokens",
                as_dict=True)["elmo"]
            sess.run(tf.global_variables_initializer())
            _emb = sess.run([embeddings])[0][0]
            for word, emb in zip(_words[0], np.array(_emb)):
                fw.write('{} {}\n'.format(word, ' '.join([str(_) for _ in emb])))
            i += batch_size
    fw.close()

def _run_w2v():

    words = [word.strip().split()[0] for word in open(vocab_file).readlines()]
    fw = open(out_file, 'w')
    fw.write('{} {}\n'.format(len(words), emb_dim))
    added_words = []
    with open(embedding_file) as f:
        for line in f:
            word, vect = line.strip().split(' ', 1)
            if word.lower() in words and word.lower() not in added_words:
                added_words.append(word.lower())
                fw.write("{} {}\n".format(word.lower(), vect))
    fw.close()

if __name__== "__main__":
    if exp_name == "elmo": _run_elmo()
    else: _run_w2v()