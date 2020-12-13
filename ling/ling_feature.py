import numpy as np
import nltk
from collections import Counter, defaultdict


emoticons = set(
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

class LingFeatureMaker():
    def __init__(self, formal_corpus_path, informal_corpus_path):
        self.formal_dict = defaultdict(lambda : 1)
        self.informal_dict = defaultdict(lambda : 1)
        
        if formal_corpus_path is not None:
            with open(formal_corpus_path) as f:
                temp = [nltk.tokenize.word_tokenize(line) for line in f.read().splitlines()]
                temp = [item for sublist in temp for item in sublist]
                formal_dict = Counter(temp)
                for word, count in formal_dict:
                    self.formal_dict[word] = count + 1

        else:
            self.formal_dict = None
        
        if informal_corpus_path is not None:
            with open(informal_corpus_path) as f:
                temp = [nltk.tokenize.word_tokenize(line) for line in f.read().splitlines()]
                temp = [item for sublist in temp for item in sublist]
                informal_dict = Counter(temp)
                for word, count in informal_dict:
                    self.informal_dict[word] = count + 1

        else:
            self.informal_dict = None

        if self.formal_dict is not None and self.informal_dict is not None:
            self.N = np.sum([count for _, count in self.informal_dict.items()]) / np.sum([count for _, count in self.formal_dict.items()])

    def word_counts(self, sent):
        def _ratio(word):
            return 2. * self.formal_dict[word] / (self.N * self.informal_dict[word] + self.formal_dict[word])

    def word_length(self, sent, L=28):
        def _FS(word):
            return 2. * np.log(len(word.split())) / np.log(L) - 1.

        return np.array([_FS(x) for x in nltk.tokenize.word_tokenize(sent)])

    def latinate(self, sent):
        def _has_prefix(word):
            return any([word.startswith(x) for x in LATIN_PREFIX])

        def _has_suffix(word):
            return any([word.endswith(x) for x in LATIN_SUFIX])
        return np.array([_has_suffix(x) or _has_prefix(x) for x in nltk.tokenize.word_tokenize(sent)])


class LingModel():
    def __init__(self):
        return

