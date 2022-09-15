import warnings
import string
from nltk.tokenize import  word_tokenize,sent_tokenize
import re
warnings.filterwarnings('ignore')

raw_docs = []

import nltk

# step1:转小写
raw_docs = [doc.lower() for doc in raw_docs]
print(raw_docs)

# step2:分词
tokenized_docs = [word_tokenize(doc) for doc in raw_docs]
print(tokenized_docs)

sent_token = [sent_tokenize(doc) for doc in raw_docs]

# step3:去除标点
regex = re.compile('[%s]' % re.escape(string.punctuation))

tokenized_docs_no_punctuation = []

for review in tokenized_docs:
    new_review = []
    for token in review:
        new_token = regex.sub(u'',token)
        if not new_token == u'':
            new_review.append(new_token)
    tokenized_docs_no_punctuation.append(new_review)

print(tokenized_docs_no_punctuation)

# step4:去除停顿词