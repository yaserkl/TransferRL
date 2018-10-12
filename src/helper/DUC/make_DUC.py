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

import os, sys
import json
import argparse
import glob
import re
import nltk.data
from nltk.tokenize.treebank import TreebankWordTokenizer
from tensorflow.core.example import example_pb2
import struct
#@lint-avoid-python-3-compatibility-imports

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
tokenizer = TreebankWordTokenizer()
def main(arguments):

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=
                                     argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sum_docs', help="Article directory.", type=str)
    parser.add_argument('--year', help="DUC year to process.", type=str)
    parser.add_argument('--result_docs', help="Reference directory.", type=str)
    parser.add_argument('--ref_dir',
                        help="Directory to output the references.", type=str)
    parser.add_argument('--sys_dir',
                        help="Directory to output the references.", type=str)
    parser.add_argument('--article_file',
                        help="File to output the article sentences..", type=str)
    parser.add_argument('--root_dir',
                        help="Directory to store bin files.", type=str)
    args = parser.parse_args(arguments)

    refs = [open("{0}/task1_ref{1}.txt".format(args.ref_dir, i), "w")
            for i in range(4)]
    article = open(args.article_file, "w")
    prefix = open(args.sys_dir + "/task1_prefix.txt", "w")
    if args.year == "2003":
        files = glob.glob("{0}/*/*".format(args.sum_docs))
    else:
        files = glob.glob("{0}/*/*".format(args.sum_docs))
    files.sort()
    writer = open('{}/test.bin'.format(args.root_dir), 'wb')
    if not os.path.exists('{}/files'.format(args.root_dir)):
        os.makedirs('{}/files/articles'.format(args.root_dir))
        os.makedirs('{}/files/abstracts'.format(args.root_dir))
        os.makedirs('{}/files/jsons/test'.format(args.root_dir))

    for cnt, f in enumerate(files):
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
            if abstract.endswith('.'): abstract[0:-1] # remove .
            abstract = '{} .'.format(abstract) # add . with one space
            print >>refs[i], abstract
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
            print >>article
            print >>prefix
            continue
        #first = sents[0]

        # If the sentence is too short, add the second as well.
        #if len(sents[0]) < 130 and len(sents) > 1:
        #    first = first.strip()[:-1] + " , " + sents[1]

        #first = " ".join(tokenizer.tokenize(first.lower()))
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

        print >>article, processed[0]
        print >>prefix, processed[0][:75]

        tf_example = example_pb2.Example() # compatible with pointer-generator model
        final_article = ' '.join(processed[0:-1]) # last line is junk
        tf_example.features.feature['article'].bytes_list.value.extend([final_article.encode()])
        processed_abstracts = ' '.join(['<s> {} </s>'.format(_) for _ in abstracts])
        tf_example.features.feature['abstract'].bytes_list.value.extend([processed_abstracts.encode()])
        tf_example_str = tf_example.SerializeToString()
        str_len = len(tf_example_str)
        writer.write(struct.pack('q', str_len))
        writer.write(struct.pack('%ds' % str_len, tf_example_str))

        with open('{}/files/articles/{}.txt'.format(args.root_dir, cnt), 'w') as faw:
            faw.write(final_article)
        with open('{}/files/abstracts/{}.txt'.format(args.root_dir, cnt), 'w') as faw:
            faw.write('\n'.join([_ for _ in abstracts]))

        json_dump = {"id":str(i), "article":processed[0:-1], "abstract": abstracts} # compatible with fast_rl_model
        with open('{}/files/jsons/test/{}.json'.format(args.root_dir, cnt), 'w') as faw:
            json.dump(json_dump, faw)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
