import pandas as pd
from pykospacing import spacing

q = pd.read_csv('data/total.csv', engine='python', encoding='utf-8', names=['sentences','labels'])[1:]

sentences = q.sentences
labels = q.labels

s, l = [], []
for sent,label in zip(sentences, labels):
    s.append(spacing(sent))
    l.append(label)

q_col = pd.DataFrame(q, columns=['sentences'])
l_col = pd.DataFrame(l, columns=['labels'])

result = pd.concat([q_col, l_col], axis=1)

result[1:].to_csv('data/tokenized.csv', index=False, encoding='utf-8-sig')