import numpy as np
import nltk
from collections import Counter, defaultdict
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline

EMOTICONS = set(
r"""
:)
:-)
:))
:-))
:)))
:-)))
(:
(-:
=)
(=
")
:]
:-]
[:
[-:
[=
=]
:o)
(o:
:}
:-}
8)
8-)
(-8
;)
;-)
(;
(-;
:(
:-(
:((
:-((
:(((
:-(((
):
)-:
=(
>:(
:')
:'-)
:'(
:'-(
:/
:-/
=/
=|
:|
:-|
]=
=[
:1
:P
:-P
:p
:-p
:O
:-O
:o
:-o
:0
:-0
:()
>:o
:*
:-*
:3
:-3
=3
:>
:->
:X
:-X
:x
:-x
:D
:-D
;D
;-D
=D
xD
XD
xDD
XDD
8D
8-D
^_^
^__^
^___^
>.<
>.>
<.<
._.
;_;
-_-
-__-
v.v
V.V
v_v
V_V
o_o
o_O
O_o
O_O
0_o
o_0
0_0
o.O
O.o
O.O
o.o
0.0
o.0
0.o
@_@
<3
<33
<333
</3
(^_^)
(-_-)
(._.)
(>_<)
(*_*)
(¬_¬)
ಠ_ಠ
ಠ︵ಠ
(ಠ_ಠ)
¯\(ツ)/¯
(╯°□°）╯︵┻━┻
><(((*>
""".split())


LATIN_PREFIX = 're,un,de,mis'.split(',')
LATIN_SUFIX  = 'ize,en,ify,le,ate,ee,er,ant,age,ment,able,ful,less,ness'.split(',')

TAGS = ['LS', 'TO', 'VBN', "''", 'WP', 'UH', 'VBG', 'JJ', 'VBZ', '--', 'VBP', 'NN', 'DT', 'PRP', ':', 'WP$', 'NNPS', 'PRP$', 'WDT', '(', ')', '.', ',', '``', '$', 'RB', 'RBR', 'RBS', 'VBD', 'IN', 'FW', 'RP', 'JJR', 'JJS', 'PDT', 'MD', 'VB', 'WRB', 'NNP', 'EX', 'NNS', 'SYM', 'CC', 'CD', 'POS']

class LingFeatureMaker():
    def __init__(self, formal_corpus_path, informal_corpus_path, use_pos_lm=False, use_lm=False):
        self.formal_dict = defaultdict(lambda : 1)
        self.informal_dict = defaultdict(lambda : 1)
        
        with open(formal_corpus_path) as f:
            formal_text = [nltk.tokenize.word_tokenize(line) for line in f.read().splitlines()]
            temp = [item for sublist in formal_text for item in sublist]
            formal_dict = Counter(temp)
            for word, count in formal_dict.items():
                self.formal_dict[word] = count + 1

        with open(informal_corpus_path) as f:
            temp = [nltk.tokenize.word_tokenize(line) for line in f.read().splitlines()]
            temp = [item for sublist in temp for item in sublist]
            informal_dict = Counter(temp)
            for word, count in informal_dict.items():
                self.informal_dict[word] = count + 1

        if self.formal_dict is not None and self.informal_dict is not None:
            self.N = np.sum([count for _, count in self.informal_dict.items()]) / np.sum([count for _, count in self.formal_dict.items()])

        self.pos_dict = {}
        for id_, tag in enumerate(TAGS):
            self.pos_dict[tag] = id_

        if use_pos_lm:
            pos_text = []
            for text in formal_text:
                pos_text.append([x[1] for x in nltk.pos_tag(text)])
            self.pos_lm = self.train_pos_lm(pos_text)

    def train_pos_lm(self, pos_text):
        train_data, padded_sents = padded_everygram_pipeline(3, pos_text)
        lm_model = MLE(3)
        lm_model.fit(train_data, padded_sents)
        return lm_model

    def word_counts(self, sent):
        def _ratio(word):
            return 2. * self.formal_dict[word] / (self.N * self.informal_dict[word] + self.formal_dict[word])

        return np.array([_ratio(x) for x in nltk.tokenize.word_tokenize(sent)])

    def word_length(self, sent, L=28):
        def _FS(word):
            return 2. * np.log(len(word.split())) / np.log(L) - 1.

        return np.array([_FS(x) for x in nltk.tokenize.word_tokenize(sent)])

    def latinate(self, sent):
        def _has_prefix(word):
            return any([word.startswith(x) for x in LATIN_PREFIX])

        def _has_suffix(word):
            return any([word.endswith(x) for x in LATIN_SUFIX])
        
        return np.array([_has_suffix(x) or _has_prefix(x) for x in nltk.tokenize.word_tokenize(sent)]).astype(float)

    def has_emoticon(self, sent):
        return float(any([x in sent for x in EMOTICONS]))

    def has_emoticon_seq(self, sent):
        return np.array(any([x in sent for x in EMOTICONS])).astype('float')

    def pos_tags(self, sent):
        return [x[1] for x in nltk.pos_tag(nltk.tokenize.word_tokenize(sent))]

    def embed_pos(self, sent):
        x = np.zeros(len(self.pos_dict))
        pos_tags = [x[1] for x in nltk.pos_tag(nltk.tokenize.word_tokenize(sent))]
        return self.pos_lm.score(pos_tags)



def make_model_inp(formal_data_path, informal_data_path):
    ling_maker = LingFeatureMaker(formal_corpus_path=formal_data_path, informal_corpus_path=informal_data_path, use_pos_lm=True)
    x = []
    y = []
    with open(formal_data_path) as f:
        for line in f:
            line = line.strip()
            x1 = np.mean(ling_maker.word_counts(line))
            x2 = np.mean(ling_maker.word_length(line))
            x3 = np.mean(ling_maker.latinate(line))
            x4 = ling_maker.has_emoticon(line)
            x5 = ling_maker.embed_pos(line)
            x.append(np.concatenate([x1, x2, x3, x4, x5]))
            y.append(1)

    with open(informal_data_path) as f:
        for line in f:
            line = line.strip()
            x1 = np.mean(ling_maker.word_counts(line))
            x2 = np.mean(ling_maker.word_length(line))
            x3 = np.mean(ling_maker.latinate(line))
            x4 = ling_maker.has_emoticon(line)
            x5 = ling_maker.embed_pos(line)
            x.append(np.concatenate([x1, x2, x3, x4, x5]))
            y.append(0)
    x = np.stack(x, axis=0)
    y = np.stack(y)
    return x, y

def run_ling_model(formal_data_path, informal_data_path):
    return
    


if __name__ == '__main__':

    formal_train_path, informal_train_path, formal_val_path, informal_val_path, formal_test_path, informal_test_path \
        = sys.args[1], sys.args[2], sys.args[3], sys.args[4], sys.args[5], sys.args[6]
    x_train, y_train = make_model_inp(formal_train_path, informal_train_path)
    x_val, y_val = make_model_inp(formal_val_path, informal_val_path)
    x_test, y_test = make_model_inp(formal_test_path, informal_test_path)
    model = LogisticRegression(random_state=0, max_iter=5000, penalty='l2').fit(x_train, y_train)
    val_acc = lr_model.score(x_val, y_val)

    print('valid score is {}'.format(val_acc))
    test_acc = lr_model.score(x_test, y_test)
    print('test score is {}'.format(test_acc))
