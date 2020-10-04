'''
File: tokenization.py
Author: Vishaal Yalamanchali
Purpose: Created to test pickling and importing serialized machine learning models.

'''
import re
import emoji
import nltkmodels
from nltkmodels import stopwords, nltk


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
    tokens = list(texts) 
    for i in tokens:
        if i not in stopwords:
            ntokens.append(i)

    return ' '.join(ntokens)  



