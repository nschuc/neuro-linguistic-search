from collections import Counter

import glob
import codecs
import random
import struct
import csv
import pandas as pd
import numpy as np
import os
import torch
from torch.autograd import Variable
import random
import pickle
import time
# <s> and </s> are used in the data files to segment the abstracts into sentences. They don't receive vocab ids.

PAD = 'PAD' #  This has a vocab id, which is used to represent out-of-vocabulary words [0]
UNK = 'UNK' #  This has a vocab id, which is used to represent out-of-vocabulary words [1]
KEEP = 'KEEP' # This has a vocab id, which is used for copying from the source [2]
DEL = 'DEL' # This has a vocab id, which is used for deleting the corresponding word [3]
START = 'START' # this has a vocab id, which is uded for indicating start of the sentence for decoding [4]
STOP = 'STOP' # This has a vocab id, which is used to stop decoding [5]

PAD_ID = 0 #  This has a vocab id, which is used to represent out-of-vocabulary words [0]
UNK_ID = 1 #  This has a vocab id, which is used to represent out-of-vocabulary words [1]
KEEP_ID = 2 # This has a vocab id, which is used for copying from the source [2]
DEL_ID = 3 # This has a vocab id, which is used for deleting the corresponding word [3]
START_ID = 4 # this has a vocab id, which is uded for indicating start of the sentence for decoding [4]
STOP_ID = 5 # This has a vocab id, which is used to stop decoding [5]

def sent2id(sent,vocab):
    """
    this function transfers a sentence (in list of strings) to an np_array
    :param sent: sentence in list of strings
    :param vocab: vocab object
    :return: sentence in numeric token numbers
    """
    new_sent = np.array([[vocab.w2i[i] if i in vocab.w2i.keys() else vocab.w2i[UNK] for i in sent]])
    return new_sent

def id2edits(ids,vocab):
    """
    #     this function transfers a id sentences of edits to actual edit actions
    #     :param ids: list of ids indicating edits
    #     :param vocab: vocab object
    #     :return: list of actual edits
    #     """
    edit_list = [vocab.i2w[i] for i in ids]
    return edit_list


def batchify(data, max_len=100): #max_len cutout defined by human
    bsz = len(data)
    try:
        maxlen_data = max([s.shape[0] for s in data])
    except:
        maxlen_data = max([len(s) for s in data])
    maxlen = min(maxlen_data, max_len)
    batch = np.zeros((bsz, maxlen), dtype=np.int)
    for i, s in enumerate(data):
        try:
            batch[i, :min(s.shape[0],maxlen)] = s[:min(s.shape[0],maxlen)]
        except:
            batch[i, :min(len(s), maxlen)] = s[:min(len(s), maxlen)]
        # batch[i, s.shape[0]:] = 3
    return Variable(torch.from_numpy(batch)).cuda()


def batchify_start_stop(data, max_len=100, start_id=4, stop_id=5): #max_len cutout defined by human
    # add start and stop tokens
    data = [np.append(s, [stop_id]) for s in data]  # stop 3
    data = [np.insert(s, 0, start_id) for s in data]  # stop 3

    bsz = len(data)
    maxlen_data = max([s.shape[0] for s in data])
    maxlen = min(maxlen_data, max_len)
    batch = np.zeros((bsz, maxlen), dtype=np.int)
    for i, s in enumerate(data):
        batch[i, :min(s.shape[0], maxlen)] = s[:min(s.shape[0],maxlen)]
        # batch[i, s.shape[0]:] = 3
    return Variable(torch.from_numpy(batch)).cuda()


def batchify_stop(data, max_len=100, start_id=4, stop_id=5): #max_len cutout defined by human
    # add start and stop tokens
    data = [np.append(s, [stop_id]) for s in data]  # stop 3

    bsz = len(data)
    maxlen_data = max([s.shape[0] for s in data])
    # print(f"dbg: maxlen_data={maxlen_data}")
    # print(f"dbg: max_len={max_len}")
    maxlen = min(maxlen_data, max_len)
    batch = np.zeros((bsz, maxlen), dtype=np.int)
    for i, s in enumerate(data):
        batch[i, :min(s.shape[0],maxlen)] = s[:min(s.shape[0],maxlen)]
        # batch[i, s.shape[0]:] = 3
    return Variable(torch.from_numpy(batch)).cuda()

class Vocab():
    def __init__(self):
        self.word_list = [PAD, UNK, KEEP, DEL, START, STOP]
        self.w2i = {}
        self.i2w = {}
        self.count = 0
        self.embedding = None

    def add_vocab_from_file(self, vocab_file="../data/vocab.txt", vocab_size=30000):
        with open(vocab_file, "r", encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= vocab_size:
                    break
                self.word_list.append(line.split()[0])  # only want the word, not the count
        print("** read %d words from vocab file" % len(self.word_list))

        for w in self.word_list:
            self.w2i[w] = self.count
            self.i2w[self.count] = w
            self.count += 1

    def add_embedding(self, gloveFile, embed_size):
        print("** Loading Glove embeddings")
        with codecs.open(gloveFile, 'r', encoding='utf-8') as f:
            model = {}
            w_set = set(self.word_list)
            embedding_matrix = np.zeros(shape=(len(self.word_list), embed_size))

            for line in f:
                splitLine = line.split()
                word = splitLine[0]
                if word in w_set:  # only extract embeddings in the word_list
                    embedding = np.array([float(val) for val in splitLine[1:]])
                    model[word] = embedding
                    embedding_matrix[self.w2i[word]] = embedding
                    # if len(model) % 1000 == 0:
                        # print("processed %d data" % len(model))
        self.embedding = embedding_matrix
        print("** %d words out of %d has embeddings in the glove file" % (len(model), len(self.word_list)))

class POSvocab():
    def __init__(self, pos_path):
        self.word_list = [PAD,UNK,START,STOP]
        self.w2i = {}
        self.i2w = {}
        self.count = 0
        self.embedding = None
        with open(pos_path, 'rb') as f:
            tagdict = pickle.load(f)

        for w in self.word_list:
            self.w2i[w] = self.count
            self.i2w[self.count] = w
            self.count += 1

        for w in tagdict:
            self.w2i[w] = self.count
            self.i2w[self.count] = w
            self.count += 1


class Datachunk():
    def __init__(self,data_path):
        self.data_path = data_path
        self.listdir = os.listdir(self.data_path)
        random.shuffle(self.listdir)
        self.idx_count = 0

    def example_generator(self,shuffle=True):
        while len(self.listdir) != 0:
            print("reading a new chunk with %d chunks remaining" % len(self.listdir))
            df = pd.read_pickle(self.data_path + self.listdir.pop())

            if shuffle:
                df = df.sample(frac=1).reset_index(drop=True)
                print('shuffling the df')

            for index, row in df.iterrows():
                self.idx_count+=1
                yield self.idx_count, row

    def batch_generator(self, batch_size=1, shuffle=True):
        while len(self.listdir) != 0:
            # print("reading a new chunk with %d chunks remaining" % len(self.listdir))
            df = pd.read_pickle(self.data_path + self.listdir.pop())

            if shuffle:
                df = df.sample(frac=1).reset_index(drop=True)
                # print('shuffling the df')

            list_df = [df[i:i + batch_size] for i in range(0, df.shape[0], batch_size)]
            for df in list_df:
                self.idx_count += 1
                yield self.idx_count, df


class Dataset():
    def __init__(self,data_path):
        self.df = pd.read_pickle(data_path)
        self.idx_count = 0

    def example_generator(self):
        for index, row in self.df.iterrows():
            yield index, row

    def batch_generator(self, batch_size=64, shuffle=True):
        if shuffle:
            df = self.df.sample(frac=1).reset_index(drop=True)
            print('* shuffling the df')

        # filter sentence with length <= 1
        print('* {} examples before length filtering'.format(self.df.shape[0]))
        self.df = self.df[self.df.comp_tokens.str.split().str.len() > 1]
        self.df = self.df[self.df.comp_tokens.str.split().str.len() <= 20]
        print('* {} examples after length filtering'.format(self.df.shape[0]))

        list_df = [self.df[i:i + batch_size] for i in range(0, self.df.shape[0], batch_size)]

        for df in list_df:
            self.idx_count += 1
            yield self.idx_count, df

def prepare_example(example,vocab):
    """
    :param example: one row in pandas dataframe with feild ['comp_tokens', 'comp_parse', 'simp_tokens', 'simp_parse', 'scpn_tokens', 'edit_labels']
    :param vocab: vocab object for translation
    :return: inp: original input sentence, syn: syntactic transformed sentence, tgt: target sentence in terms of edit operations with respect to syn
    """
    inp = np.array([vocab.w2i[i] if i in vocab.w2i.keys() else vocab.w2i[UNK] for i in example['comp_tokens']])
    syn = np.array([vocab.w2i[i] if i in vocab.w2i.keys() else vocab.w2i[UNK] for i in  example['scpn_tokens']])
    tgt = [vocab.w2i[i] if i in vocab.w2i.keys() else vocab.w2i[UNK] for i in  example['edit_labels']]
    # pad START and STOP to tgt
    tgt = np.array([vocab.w2i[START]] +tgt)
    # print(inp,syn,tgt)
    return inp,syn,tgt # add a dimension for batch, batch_size =1


def batchify_start_stop_tgt(data, max_len=100,start_id=4,stop_id=5): #max_len cutout defined by human
    def find_mask(edit_seq):
        mask_s = np.zeros(len(edit_seq))
        for i in range(len(edit_seq))[::-1][1:]: #skip stop, look from the back
            if edit_seq[i] == KEEP_ID:
                if edit_seq[i-1] ==KEEP_ID:
                    mask_s[i-1]=1
            #mask all keep
            # if edit_seq[i] == KEEP_ID:
            #         mask_s[i]=1
            else:
                break
        return mask_s
    # add start and stop tokens
    data = [np.append(s, [stop_id]) for s in data]  # stop 3
    data = [np.insert(s, 0, start_id) for s in data]  # stop 3

    bsz = len(data)
    maxlen_data = max([s.shape[0] for s in data])
    maxlen = min(maxlen_data, max_len)
    batch = np.zeros((bsz, maxlen), dtype=np.int)
    mask = np.zeros((bsz, maxlen), dtype=np.int)  # mask out keep before the end
    for i, s in enumerate(data):

        batch[i, :min(s.shape[0],maxlen)] = s[:min(s.shape[0],maxlen)]
        mask_s = find_mask(s)
        mask[i,:min(s.shape[0],maxlen)]=mask_s[:min(s.shape[0],maxlen)]
        # batch[i, s.shape[0]:] = 3
    return Variable(torch.from_numpy(batch)).cuda(),torch.from_numpy(mask)

def prepare_batch(batch_df, max_length=100):
    # time1 = time.time()
    # result = batch_df.apply(lambda x:prepare_example(x,vocab),axis=1)
    # time2 = time.time()
    # print("example_prep_time", time2 - time1)
    # results=zip(*result)
    # print(batch_df.columns.values)
    inp = batchify_stop(batch_df['comp_ids'], max_len=max_length)
    inp_pos = batchify_stop(batch_df['comp_pos_ids'], max_len=max_length)
    inp_simp = batchify_start_stop(batch_df['simp_id'], max_len=max_length)
    tgt = batchify_start_stop(batch_df['new_edit_ids'], max_len=max_length)  # edit sequence is usually longer

    return [inp, inp_pos, tgt, inp_simp], batch_df['comp_tokens'], batch_df['simp_tokens']


# def prepare_batch(batch_df,vocab, max_length=100):
#     inp_list,syn_list,tgt_list,syn_tokens_list=[],[],[],[]
#     for index, example in batch_df.iterrows():
#         result = prepare_example(example,vocab)
#         inp_list.append(result[0])
#         syn_list.append(result[1])
#         tgt_list.append(result[2])
#         syn_tokens_list.append(example['scpn_tokens'])
#     inp = batchify(inp_list,max_len=max_length)
#     syn = batchify(syn_list,max_len=max_length+50)
#     tgt = batchify(tgt_list,max_len=max_length)
#     return [inp,syn,tgt],syn_tokens_list



def article2ids(article_words, vocab):
    """Map the article words to their ids. Also return a list of OOVs in the article.

    Args:
        article_words: list of words (strings)
        vocab: Vocabulary object

    Returns:
        ids:
            A list of word ids (integers); OOVs are represented by their temporary article OOV number. If the vocabulary size is 50k and the article has 3 OOVs, then these temporary OOV numbers will be 50000, 50001, 50002.
        oovs:
            A list of the OOV words in the article (strings), in the order corresponding to their temporary article OOV numbers."""
    ids = []
    oovs = []
    unk_id = vocab.word2id(UNK)
    for w in article_words:
        i = vocab.word2id(w)
        if i == unk_id: # If w is OOV
            if w not in oovs: # Add to list of OOVs
                oovs.append(w)
            oov_num = oovs.index(w) # This is 0 for the first article OOV, 1 for the second article OOV...
            ids.append(vocab.size() + oov_num) # This is e.g. 50000 for the first article OOV, 50001 for the second...
        else:
            ids.append(i)
    return ids, oovs


def abstract2ids(abstract_words, vocab, article_oovs):
    """Map the abstract words to their ids. In-article OOVs are mapped to their temporary OOV numbers.

    Args:
        abstract_words: list of words (strings)
        vocab: Vocabulary object
        article_oovs: list of in-article OOV words (strings), in the order corresponding to their temporary article OOV numbers

    Returns:
        ids: List of ids (integers). In-article OOV words are mapped to their temporary OOV numbers. Out-of-article OOV words are mapped to the UNK token id."""
    ids = []
    unk_id = vocab.word2id(UNK)
    for w in abstract_words:
        i = vocab.word2id(w)
        if i == unk_id: # If w is an OOV word
            if w in article_oovs: # If w is an in-article OOV
                vocab_idx = vocab.size() + article_oovs.index(w) # Map to its temporary article OOV number
                ids.append(vocab_idx)
            else: # If w is an out-of-article OOV
                ids.append(unk_id) # Map to the UNK token id
        else:
            ids.append(i)
    return ids

def outputids2words(id_list, vocab, article_oovs):
    """Maps output ids to words, including mapping in-article OOVs from their temporary ids to the original OOV string (applicable in pointer-generator mode).

    Args:
        id_list: list of ids (integers)
        vocab: Vocabulary object
        article_oovs: list of OOV words (strings) in the order corresponding to their temporary article OOV ids (that have been assigned in pointer-generator mode), or None (in baseline mode)

    Returns:
        words: list of words (strings)
    """
    words = []
    for i in id_list:
        try:
            w = vocab.id2word(i) # might be [UNK]
        except ValueError as e: # w is OOV
            assert article_oovs is not None, "Error: model produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
            article_oov_idx = i - vocab.size()
            #print(article_oov_idx)
            try:
                #print(article_oovs)
                w = article_oovs[article_oov_idx]
            except ValueError as e: # i doesn't correspond to an article oov
                raise ValueError('Error: model produced word ID %i which corresponds to article OOV %i but this example only has %i article OOVs' % (i, article_oov_idx, len(article_oovs)))
        words.append(w)
    return words
'''

def abstract2sents(abstract):
    """Splits abstract text from datafile into list of sentences.

    Args:
        abstract: string containing <s> and </s> tags for starts and ends of sentences

    Returns:
        sents: List of sentence strings (no tags)"""
    cur = 0
    sents = []
    while True:
        try:
            start_p = cur#abstract.index( start_tok, cur)
            end_p = abstract.index('.', start_p + 1)
            cur = end_p + 1
            sents.append(abstract[start_p:cur])
        except ValueError as e:
            return sents



def show_art_oovs(article, vocab):
    """Returns the article string, highlighting the OOVs by placing __underscores__ around them"""
    unk_token = vocab.word2id(UNKNOWN_TOKEN)
    words = article.split(' ')
    words = [("__%s__" % w) if vocab.word2id(w)==unk_token else w for w in words]
    out_str = ' '.join(words)
    return out_str


def show_abs_oovs(abstract, vocab, article_oovs):
    """Returns the abstract string, highlighting the article OOVs with __underscores__.

    If a list of article_oovs is provided, non-article OOVs are differentiated like !!__this__!!.

    Args:
        abstract: string
        vocab: Vocabulary object
        article_oovs: list of words (strings), or None (in baseline mode)
    """
    unk_token = vocab.word2id(UNKNOWN_TOKEN)
    words = abstract.split(' ')
    new_words = []
    for w in words:
        if vocab.word2id(w) == unk_token: # w is oov
            if article_oovs is None: # baseline mode
                new_words.append("__%s__" % w)
            else: # pointer-generator mode
                if w in article_oovs:
                    new_words.append("__%s__" % w)
                else:
                    new_words.append("!!__%s__!!" % w)
        else: # w is in-vocab word
            new_words.append(w)
    out_str = ' '.join(new_words)
    return out_str
'''
