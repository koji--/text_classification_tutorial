from os.path import normpath, dirname, join
import MeCab
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from pprint import pprint


tagger = MeCab.Tagger()
tagger.parse('')

def tokenize(text):
    node = tagger.parseToNode(text)

    tokens = []
    while node:
        if node.surface != '':
            tokens.append(node.surface)

        node = node.next

    return tokens

# データ読み込み
BASE_DIR = normpath(dirname(__file__))
csv_path = join(BASE_DIR, './training_data.csv')
training_data = pd.read_csv(csv_path)
texts = training_data['text']
print(texts)

# Bag of Words計算
vectorizer = CountVectorizer(analyzer=tokenize)
vectorizer.fit(texts)
bow = vectorizer.transform(texts)
print(bow.toarray())
