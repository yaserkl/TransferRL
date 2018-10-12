#
#  Copyright (c) 2015, Facebook, Inc.
#  All rights reserved.
#
#  This source code is licensed under the BSD-style license found in the
#  LICENSE file in the root directory of this source tree. An additional grant
#  of patent rights can be found in the PATENTS file in the same directory.
#
#  Author: Alexander M Rush <srush@seas.harvard.edu>
#          Sumit Chopra <spchopra@fb.com>
#          Jason Weston <jase@fb.com>

"""Construct the DUC test set. """

#python src/helper/DUC/make_DUC_train.py --year 2004 --root_dir /home/yaserkl/data/DUC/clean_2004/splits/ --sum_docs /home/yaserkl/data/DUC/DUC2004_Summarization_Documents/duc2004_testdata/tasks1and2/duc2004_tasks1and2_docs/docs/ --result_docs /home/yaserkl/data/DUC/duc2004_results/ROUGE/eval/models/1/
#python src/helper/DUC/make_DUC_train.py --year 2003 --root_dir /home/yaserkl/data/DUC/clean_2003/splits/ --sum_docs /home/yaserkl/data/DUC/DUC2003_Summarization_Documents/duc2003_testdata/task1/docs.without.headlines/ --result_docs /home/yaserkl/data/DUC/detagged.duc2003.abstracts/models/

import os, sys
import argparse
import glob
import re
import nltk.data
from nltk.tokenize.treebank import TreebankWordTokenizer
from tensorflow.core.example import example_pb2
import struct
from random import shuffle

#@lint-avoid-python-3-compatibility-imports

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
tokenizer = TreebankWordTokenizer()
def main(arguments):

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=
                                     argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sum_docs', help="Article directory.", type=str)
    parser.add_argument('--result_docs', help="Reference directory.", type=str)
    parser.add_argument('--year', help="DUC year to process.", type=str)
    parser.add_argument('--root_dir',
                        help="Directory to store bin files.", type=str)
    args = parser.parse_args(arguments)

    if args.year == "2003":
        files = glob.glob("{0}/*/*".format(args.sum_docs))
    else:
        files = glob.glob("{0}/*/*".format(args.sum_docs))
    files.sort()
    shuffle(files)
    train_ind = int(len(files)*0.2)
    val_ind = int(len(files)*0.1)
    train_files = files[0:train_ind]
    val_files = files[train_ind:(train_ind+val_ind)]
    test_files = files[(train_ind+val_ind):]
    print(len(train_files), len(val_files), len(test_files))

    if not os.path.exists(args.root_dir):
        os.makedirs(args.root_dir)

    for fls, ds_name in zip([train_files, val_files, test_files], ['train', 'val', 'test']):
        writer = open('{}/{}.bin'.format(args.root_dir, ds_name), 'wb')
        for cnt, f in enumerate(fls):
            docset = f.split("/")[-2][:-1].upper()
            name = f.split("/")[-1].upper()

            # Find references.
            if args.year == "2003":
                matches = list(glob.glob("{0}/{1}*.10.*{2}*".format(
                    args.result_docs, docset, name)))
            else:
                matches = list(glob.glob("{0}/{1}*{2}*".format(
                    args.result_docs, docset, name)))
            matches.sort()
            assert len(matches) == 4, matches
            abstracts = []
            for i, m in enumerate(matches):
                abstract = open(m).read().strip()
                if abstract.endswith('.'): abstract = abstract[0:-1] # remove .
                abstract = '{} .'.format(abstract) # add . with one space
                abstracts.append(abstract)

            # Make input.
            mode = 0
            text = ""
            for l in open(f):
                if l.strip() in ["</P>", "<P>"]:
                    continue
                if mode == 1 and l.strip() != "<P>":
                    text += l.strip() + " "
                if l.strip() == "<TEXT>":
                    mode = 1
            text = " ".join([w for w in text.split() if w[0] != "&"])

            sents = sent_detector.tokenize(text)
            if len(sents) == 0:
                continue

            processed = []
            for sent in sents:
                sent = sent.strip()
                sent = " ".join(tokenizer.tokenize(sent.lower()))
                if ")" in sent or ("_" in sent and args.year == "2003"):
                    sent = re.split(" ((--)|-|_) ", sent, 1)[-1]
                sent = sent.replace("(", "-lrb-") \
                             .replace(")", "-rrb-").replace("_", ",")
                if sent.endswith('.'):
                    sent = sent[0:-1] # remove .
                sent = '{} .'.format(sent.strip()) # add . with one space
                processed.append(sent)

            tf_example = example_pb2.Example() # compatible with pointer-generator model
            final_article = ' '.join(processed[0:-1]) # last line is junk
            tf_example.features.feature['article'].bytes_list.value.extend([final_article.encode()])
            processed_abstracts = ' '.join(['<s> {} </s>'.format(_) for _ in abstracts])
            tf_example.features.feature['abstract'].bytes_list.value.extend([processed_abstracts.encode()])
            tf_example_str = tf_example.SerializeToString()
            str_len = len(tf_example_str)
            writer.write(struct.pack('q', str_len))
            writer.write(struct.pack('%ds' % str_len, tf_example_str))

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
