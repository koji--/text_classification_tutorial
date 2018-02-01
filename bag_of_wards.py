import MeCab
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

def calc_bow(tokenized_texts):
	# Build vocabulary (4)
	vocabulary = {}
	for i, tokenized_text in enumerate(tokenized_texts):
		for token in tokenized_text:
			if token not in vocabulary:
				vocabulary[token] = {
					"index":len(vocabulary),
					"count_list":[0 for k in range(len(tokenized_texts))]
				}

			vocabulary[token]["count_list"][i] += 1

	bow = [[0 for k in range(len(vocabulary))] for j in range(len(tokenized_texts))]
	for token, item in vocabulary.items():
		index = item['index']
		count_list = item["count_list"]
		for i, v in enumerate(count_list):
			bow[i][index] = v

	return vocabulary, bow

# 入力分のlist
texts = [
	'私は私のことが好きなあなたが好きです',
	'私はラーメンが好きです',
	'富士山は日本一高い山です',
]

tokenized_texts = [tokenize(text) for text in texts]
vocabulary, bow = calc_bow(tokenized_texts)
pprint(vocabulary)
pprint(bow)
