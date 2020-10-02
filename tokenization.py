'''
File: tokenization.py
Author: Vishaal Yalamanchali
Purpose: Created to test pickling and importing serialized machine learning models.

'''
import nltk
import re
import emoji
from nltk.tokenize import word_tokenize

from nltk.stem import SnowballStemmer
from nltk.tokenize import sent_tokenize

from nltk.corpus import sentiwordnet as swn
# Do this first, that'll do something eval() 
# to "materialize" the LazyCorpusLoader
next(swn.all_senti_synsets()) 
from nltk.stem import WordNetLemmatizer
lemma = WordNetLemmatizer()


stopwords = nltk.corpus.stopwords.words('english')
stopwords.remove('not')
stopwords.remove('nor')
stopwords.remove('no')

pattern = '@\S+|https?:\S+|http?:\S|[^A-Za-z]+|com|net'
urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
userPattern       = '@[^\s]+'
alphaPattern      = "[^a-zA-Z0-9]"
sequencePattern   = r"(.)\1\1+"
seqReplacePattern = r"\1\1"

def tokenize(texts):

    # Remove special chars
    texts = re.sub(r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)", ' URL',texts)
    texts = emoji.demojize(texts, delimiters=("", ""))
    texts = re.sub(userPattern,' USER', texts)
    texts = texts.replace(r"’", "'")
    texts = re.sub(alphaPattern, " ", texts)
    # Remove repetitions
    texts = re.sub(sequencePattern, seqReplacePattern, texts)

    # Transform short negation form
    texts = texts.replace(r"(can't|cannot)", 'can not')
    texts = texts.replace(r"n't", ' not')

    # Remove stop words

    tokens = []
    ntokens = []
    tokens = word_tokenize(texts) 
    for i in tokens:
        if i not in stopwords:
            ntokens.append(lemma.lemmatize(i))

    return ' '.join(ntokens) 



