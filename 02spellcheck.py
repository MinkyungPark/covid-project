import pandas as pd
from hanspell import spell_checker


q = pd.read_csv('data/tokenized.csv', engine='python', encoding='utf-8', names=['sentences','labels'])[1:]

sentences = q.sentences
labels = q.labels

s, l = [], []
for sent,label in zip(sentences, labels):
    s.append(spell_checker.check(sent).checked)
    l.append(label)



result = pd.DataFrame([x for x in zip(s,l)], columns=['sentences','labels'])

result.to_csv('data/spellchecked.csv', index=False, encoding='utf-8-sig')