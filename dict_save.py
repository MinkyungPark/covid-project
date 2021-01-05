import numpy as np
import pandas as pd

seed = 0
np.random.seed(seed)

q = pd.read_csv('data/postagged.csv', names=['sentences', 'labels'], encoding='utf-8')


sentences = q.sentences[1:]
labels = q.labels[1:]


##### sentences dictionary
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences) # 빈도수에 기반한 사전
x_data = tokenizer.texts_to_sequences(sentences)

word_index = tokenizer.word_index # word_idx 토큰의 갯수
index_word = {index+1: word for index, word in enumerate(word_index)}
word_index['OOV'] = 0
index_word[0] = 'OOV'


##### category dictionary
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

labels_set = list(set(labels))
labels_sort = sorted(labels_set)
labels_to_idx = {word: index for index, word in enumerate(labels_sort)}
idx_to_labels = {index: word for index, word in enumerate(labels_sort)}


# {'가슴통증': 0, '결막염': 1, '고열': 2, '두통': 3, '마른기침': 4, '몸살': 5, '미각이상': 6, '설사': 7, '손발가락변색': 8, '운동장애': 9, '인후통': 10, '피로감': 11, '피부발진': 12, '호흡곤란': 13, '후각이상': 14}
# {0: '가슴통증', 1: '결막염', 2: '고열', 3: '두통', 4: '마른기침', 5: '몸살', 6: '미각이상', 7: '설사', 8: '손발가락변색', 9: '운동장애', 10: '인후통', 11: '피로감', 12: '피부발진', 13: '호흡곤란', 14: '후각이상'}
##### pickle로 저장
# word_index
# index_word
# category_to_idx
# idx_to_category


# SAVE
import pickle

with open('word_index.pickle', 'wb') as fw:
    pickle.dump(word_index, fw)

with open('index_word.pickle', 'wb') as fw:
    pickle.dump(index_word, fw)

with open('label_to_idx.pickle', 'wb') as fw:
    pickle.dump(labels_to_idx, fw)

with open('idx_to_label.pickle', 'wb') as fw:
    pickle.dump(idx_to_labels, fw)



# LOAD
# with open('word_index.pickle', 'rb') as fr:
#     word_index = pickle.load(fr)

# with open('index_word.pickle', 'rb') as fr:
#     index_word = pickle.load(fr)

# with open('category_to_idx.pickle', 'rb') as fr:
#     category_to_idx = pickle.load(fr)

# with open('idx_to_category.pickle', 'rb') as fr:
#     idx_to_category = pickle.load(fr)

# print(word_index['OOV'])
# print(index_word[0])
# print(category_to_idx['두통'])
# print(idx_to_category[1])