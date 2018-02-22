from os.path import normpath, dirname, join
import MeCab
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
import pandas as pd

MECAB_DIC_DIR = '/usr/lib/mecab/dic/mecab-ipadic-neologd'


class DialogueAgent(object):
    def __init__(self):
        self.tagger = MeCab.Tagger('-d {}'.format(MECAB_DIC_DIR))
        self.tagger.parse('')  # workaround

    def _tokenize(self, text):
        node = self.tagger.parseToNode(text)

        tokens = []
        while node:
            if node.surface != '':
                tokens.append(node.surface)

            node = node.next

        return tokens

    def train(self, texts, labels):
        vectorizer = CountVectorizer(analyzer=self._tokenize)
        bow = vectorizer.fit_transform(texts)

        classifier = SVC()
        classifier.fit(bow, labels)

        self.vectorizer = vectorizer
        self.classifier = classifier

    def predict(self, texts):
        bow = self.vectorizer.transform(texts)
        return self.classifier.predict(bow)

if __name__ == '__main__':
    BASE_DIR = normpath(dirname(__file__))

training_data = pd.read_csv(join(BASE_DIR, './training_data.csv'))

dialogue_agent = DialogueAgent()
dialogue_agent.train(training_data['text'], training_data['label'])

with open(join(BASE_DIR, './replies.csv')) as f:
    replies = f.read().split('\n')

while True:
    input_text = input()
    predictions = dialogue_agent.predict([input_text])
    predicted_class_id = predictions[0]

    print(replies[predicted_class_id])
    if predicted_class_id == 3 or input_text == 'e':
        break
