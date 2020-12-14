import os
from io import open
import torch
import codecs
import numpy as np
import pandas as pd
from collections import Counter

class Vocab(object):
    def __init__(self):
        self.word_list = ['<unk>', '<blank>', '<s>', '</s>']
        self.word2idx = {}
        self.idx2word = {}
        # self.idx2word = ['<unk>', '<blank>', '<s>', '</s>']
        self.count = 0
        self.embedding = None

    def get_vocab_file(self, data_path, vocab_path, vocab_size=50000):
        df = pd.read_csv(data_path)
        sents = df.sent.tolist()
        all_words = []
        for sent in sents:
            all_words.extend(sent.strip().split())
        counter = Counter(all_words)
        topn = counter.most_common(vocab_size)

        with codecs.open(vocab_path, 'w', encoding='utf-8') as f:
            for ex in topn:
                num = str(ex[1])
                f.write(' '.join([ex[0], num]) + '\n')
        print('** saving vocab file to {}'.format(vocab_path))

    def add_vocab_from_file(self, vocab_file, vocab_size=30000):
        with codecs.open(vocab_file, "r", encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= vocab_size:
                    break
                self.word_list.append(line.split()[0])  # only want the word, not the count
        print("** read %d words from vocab file" % len(self.word_list))

        for w in self.word_list:
            self.word2idx[w] = self.count
            self.idx2word[self.count] = w
            self.count += 1

    def add_embedding(self, gloveFile, embed_size):
        print("** Loading Glove embeddings")
        with codecs.open(gloveFile, 'r', encoding='utf-8') as f:
            model = {}
            w_set = set(self.word_list) # len(w_set) == len(word_list)
            embedding_matrix = np.zeros(shape=(len(self.word_list), embed_size))

            for line in f:
                splitLine = line.split()
                word = splitLine[0]
                if word in w_set:  # only extract embeddings in the word_list
                    # print(word)
                    # print(splitLine[1:])
                    embedding = np.array([float(val) for val in splitLine[1:]])
                    model[word] = embedding
                    embedding_matrix[self.word2idx[word]] = embedding
                    # if len(model) % 1000 == 0:
                        # print("processed %d data" % len(model))
        self.embedding = torch.FloatTensor(embedding_matrix)
        print("** %d words out of %d has embeddings in the glove file" % (len(model), len(self.word_list)))

    def __len__(self):
        return len(self.word2idx)

class Corpus(object):
    def __init__(self, path, vocab):
        self.vocab = vocab
        # path = /research/king3/lijj/tesla/data/style/style_data/Family_Relationships/cls/
        self.train_src, self.train_lens, self.train_tgt = self.tokenize(os.path.join(path, 'train.tok.csv'))
        self.valid_src, self.valid_lens, self.valid_tgt = self.tokenize(os.path.join(path, 'tune.tok.csv'))
        self.test_src, self.test_lens, self.test_tgt = self.tokenize(os.path.join(path, 'test.tok.csv'))

    def tokenize(self, path, max_length=40):
        # numericalize input seq. pad seq.
        df = pd.read_csv(path, encoding='utf-8')
        idss = []
        lens = []
        for line in df.sent:
            words = line.split()[:max_length]
            while len(words) < max_length:
                words.append('<blank>')
            ids = []
            for word in words:
                if word in self.vocab.word_list:
                    ids.append(self.vocab.word2idx[word])
                else:
                    ids.append(self.vocab.word2idx['<unk>'])
            idss.append(torch.tensor(ids).type(torch.int64))
            lens.append(len(ids))
        ids = torch.stack(idss, dim=1) # max_len x num_sent
        lens = torch.tensor(lens)
        labels = torch.tensor(df.label.tolist()).type_as(ids)
        return ids.t().cuda(), lens.cuda(), labels.cuda() # num_sent x max_len, num_sent

def sent2tensor(word2idx, sents):
    max_len = max([len(x) for x in sents]) + 2 # consider the <s> </s> token
    tensor = torch.zeros(len(sents), max_len, dtype=torch.int64) + word2idx['<blank>'] # bsz x len
    for b, sent in enumerate(sents):
        sent = sent.split()[:max_len]
        tensor[b, :len(sent)] = torch.tensor(
                    [word2idx[x] if x in word2idx.keys() else word2idx['<unk>'] for x in sent]).type(torch.int64)
    return tensor.cuda()
