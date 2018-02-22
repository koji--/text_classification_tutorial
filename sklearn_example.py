import MeCab
from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer

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

texts = [
	'私は私のことが好きなあなたが好きです',
	'私はラーメンが好きです',
	'富士山は日本一高い山です',
]
print(texts)

# Bag of Words計算
vectorizer = CountVectorizer(analyzer=tokenize)
vectorizer.fit(texts)
bow = vectorizer.transform(texts)
print(bow.toarray())
