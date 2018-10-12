
# coding: utf-8

# In[1]:


from __future__ import division
import rouge
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import data
import os, sys
import struct
import pickle
from collections import defaultdict
from tensorflow.core.example import example_pb2
from unidecode import unidecode
from skipthoughts import skipthoughts
from multiprocessing import Pool, Array, cpu_count
#from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm
import spacy
import scipy.spatial.distance as sd
model = skipthoughts.load_model()
encoder = skipthoughts.Encoder(model)


# In[18]:


def custom_pipeline(nlp):
    return (nlp.sbd)

nlp = spacy.load('en')
nlp.add_pipe(nlp.create_pipe('sentencizer'))


# In[19]:


nlp.pipeline


# In[20]:


def remove_non_ascii(text):
    try:
        return unicode(unidecode(unicode(text, encoding = "utf-8")))
    except:
        return str(unidecode(text))


# In[21]:


def read_file(filepath):
    reader = open(filepath, 'rb')
    articles = []
    abstracts = []
    while True:
        len_bytes = reader.read(8)
        if not len_bytes: break # finished reading this file
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        e = example_pb2.Example.FromString(example_str)
        article = e.features.feature['article'].bytes_list.value[0] # the article text was saved under the key 'article' in the data files
        abstract = e.features.feature['abstract'].bytes_list.value[0] # the abstract text was saved under the key 'abstract' in the data files
        abstract_sentences = [sent.strip() for sent in data.abstract2sents(abstract)]
        articles.append(article)
        abstracts.append(abstract_sentences)
    return articles, abstracts


# In[15]:


def spacy_tokenizer(article):
    try:
        doc = nlp(remove_non_ascii(article))
        return [sent.text for sent in doc.sents]
    except Exception as ex:
        print("error at {}".format(article))
        return []

def manual_tokenizer(article):
    return ["{} .".format(_) for _ in remove_non_ascii(article).split(' . ')]


# In[16]:


def _run(params):
    try:
        sents, measure, replacement, abstracts = params
        sent_candidates = []
        best_sent_inds = []
        if 'skipthought' in measure:
            abstract_vectors = encoder.encode(abstracts, verbose=0)
            article_vectors = encoder.encode(sents, verbose=0)
        max_rewards = []
        for i, abst in enumerate(abstracts):
            reward_sents = []
            for _, sent in enumerate(sents):
                if not replacement and _ in best_sent_inds:
                    reward_sents.append(0)
                    continue
                if 'rouge' in measure:
                    reward_sents.append(rouge.rouge([sent], [abst])[measure])
                elif 'skipthought' in measure:
                    reward_sents.append(1 - sd.cdist([abstract_vectors[i]], article_vectors, "cosine")[0][0])
            best_sent_ind = np.argmax(reward_sents)
            best_sent_inds.append(best_sent_ind)
            sent_candidates.append(sents[best_sent_ind])
            max_rewards.append(np.max(reward_sents))
        sent_candidates = list(set(sent_candidates))
        if 'rouge' in measure:
            oracle_reward = rouge.rouge(sent_candidates, abstracts)[measure]
        elif 'skipthought' in measure:
            oracle_reward = np.sum(max_rewards)/float(len(abstracts))
    except Exception as ex:
        oracle_reward = 0
        sent_candidates = []
    return (oracle_reward, sent_candidates)

def find_closest_oracle(article_sents, abstracts, measure='rouge_l/f_score', replacement = True):
    params = [(_sents, measure, replacement, _abstracts) for (_sents, _abstracts) in zip(article_sents, abstracts) if len(_sents)>0]
    pool = Pool(cpu_count())
    results = list(tqdm(pool.imap(_run, params), total=len(article_sents)))
    pool.close()
    oracle_rewards = [_[0] for _ in results]
    sent_candidates = [_[1] for _ in results]
    return np.mean(oracle_rewards), sent_candidates


# In[ ]:


stats = defaultdict(list)
data_root = '/home/yaserkl/data'
if not os.path.exists('{}/oracle/'.format(data_root)):
    os.makedirs('{}/oracle/raw'.format(data_root))
    os.makedirs('{}/oracle/processed'.format(data_root))
df = None
if os.path.exists('{}/oracle/stats.csv'.format(data_root)):
    df = pd.DataFrame.from_csv('{}/oracle/stats.csv'.format(data_root))
    print('loaded prev_stat file with {} shape'.format(df.shape))

for dataset, dataset_path in zip(['cnn_dm', 'newsroom'],['cnn_dm/finished_files', 'tian']):
    for mode in ['test', 'val']:
        for tokenizer, tokenizer_name in zip([manual_tokenizer, spacy_tokenizer], ['manual', 'spacy']):
            articles, abstracts = read_file('{}/{}/{}.bin'.format(data_root, dataset_path, mode))
            assert len(articles)==len(abstracts)
            if os.path.exists('{}/oracle/raw/{}-{}-{}.pck'.format(data_root, dataset, mode, tokenizer_name)):
                with open('{}/oracle/raw/{}-{}-{}.pck'.format(data_root, dataset, mode, tokenizer_name), 'rb') as f:
                    article_sents = pickle.load(f)
            else:
                print('tokenizing the dataset: {}, mode: {}, tokenizer: {} ...'.format(dataset, mode, tokenizer_name))
                pool = Pool(cpu_count())
                article_sents = list(tqdm(pool.imap(tokenizer, articles), total=len(articles)))
                pool.close()
                with open('{}/oracle/raw/{}-{}-{}.pck'.format(data_root, dataset, mode, tokenizer_name), 'wb') as f:
                    pickle.dump(article_sents, file=f)
            for measure in ['rouge_1/p_score', 'rouge_1/r_score', 'rouge_1/f_score', 'rouge_2/p_score', 'rouge_2/r_score', 'rouge_2/f_score', 'rouge_l/p_score', 'rouge_l/r_score', 'rouge_l/f_score']:
                for replacement in [True, False]: # selected sentences have the same length as the summary sentences
                    if df is not None:
                        filtered_df = df[(df['dataset']== dataset) & (df['mode']== mode) & (df['tokenizer']==tokenizer_name) & (df['measure']==measure) & (df['replacement']==replacement)]
                        if filtered_df.shape[0]==1:
                            stats['dataset'].append(dataset)
                            stats['mode'].append(mode) # test, val
                            stats['tokenizer'].append(tokenizer_name) # manual, spacy
                            stats['measure'].append(measure)
                            stats['replacement'].append(replacement)
                            oracle_score = filtered_df['oracle_score'].values[0]
                            stats['oracle_score'].append(oracle_score)
                            #print('dataset {}, mode: {}, tokenizer: {}, measure: {}, replacement: {}, oracle_score: {:.3f}'.format(dataset, 
                            #                                                                                       mode, 
                            #                                                                                       tokenizer_name, 
                            #                                                                                       measure, 
                            #                                                                                       replacement, 
                            #                                                                                       oracle_score))
                            continue
                    oracle_score, candidates = find_closest_oracle(article_sents, 
                                                                   abstracts, 
                                                                   measure, 
                                                                   replacement)
                    out_dir = '{}/oracle/candidates/{}-{}-{}-{}-{}'.format(data_root, 
                                                                       dataset, 
                                                                      mode, 
                                                                      tokenizer_name, 
                                                                      measure.replace('/','_'), 
                                                                      'with_replacement' if replacement else 'no_replacement')
                    if not os.path.exists(out_dir):
                        os.makedirs('{}/decoded/'.format(out_dir))
                        os.makedirs('{}/reference/'.format(out_dir))
                    for i, (cand, abst) in enumerate(zip(candidates, abstracts)):
                        with open('{}/decoded/{}.txt'.format(out_dir, i), 'w') as f:
                            f.write(remove_non_ascii('\n'.join(cand)))
                        with open('{}/reference/{}.txt'.format(out_dir, i), 'w') as f:
                            f.write(remove_non_ascii('\n'.join(abst)))
                    print('dataset {}, mode: {}, tokenizer: {}, measure: {}, replacement: {}, oracle_score: {:.3f}'.format(dataset, 
                                                                                                                   mode, 
                                                                                                                   tokenizer_name, 
                                                                                                                   measure, 
                                                                                                                   replacement, 
                                                                                                                   oracle_score))
                    stats['dataset'].append(dataset)
                    stats['mode'].append(mode) # test, val
                    stats['tokenizer'].append(tokenizer_name) # manual, spacy
                    stats['measure'].append(measure)
                    stats['replacement'].append(replacement)
                    stats['oracle_score'].append(oracle_score)
                    df = pd.DataFrame.from_dict(stats, orient='columns')
                    df.to_csv('{}/oracle/stats.csv'.format(data_root))
                    with open('{}/oracle/processed/{}-{}-{}-{}-{}.pck'.format(data_root, 
                                                                  dataset, 
                                                                  mode, 
                                                                  tokenizer_name, 
                                                                  measure.replace('/','_'), 
                                                                  'with_replacement' if replacement else 'no_replacement'), 'wb') as f:
                        pickle.dump(candidates, f)

