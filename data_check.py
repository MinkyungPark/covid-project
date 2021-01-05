# %% 1
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import numpy as np
import pandas as pd

seed = 0
np.random.seed(seed)

df = pd.read_csv('data/postagged.csv', encoding='utf-8')
# data3 -> 1 data 1 question

print(df.info())

sentences = list(df['sentences'].astype(str))
labels = list(df['labels'].astype(str))


print(sentences[0])
print(labels[0])

print(len(sentences))
print(len(labels))

# 코로나 걸리다 가슴 통증 있다 하다 걸리다 식 아프 
# 가슴통증
# 18004
# 18004

# %% 2
# tokenize 되어있는 데이터기 때문에 따로 konlpy 사용X

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences) # 빈도수에 기반한 사전
x_data = tokenizer.texts_to_sequences(sentences)

word_index = tokenizer.word_index # word_idx 토큰의 갯수 22574개
index_word = {index+1: word for index, word in enumerate(word_index)}

print(len(word_index)) # 11121

############### DATA EXPLORE ######################
# %% 문장 길이 확인
# token x char length o
print('문장 최대 토큰 갯수 : {}'.format(max([len(sentence) for sentence in x_data])))
print('문장 평균 토큰 갯수 : {}'.format(sum(map(len, x_data))/len(x_data)))

import matplotlib.pyplot as plt

plt.hist([len(s) for s in x_data], bins=50)
plt.xlabel('length of sentences')
plt.ylabel('number of sentences')
plt.show()

# 문장 최대 토큰 갯수 : 164
# 문장 평균 토큰 갯수 : 43.64774494556765

# %% 각 카테고리 당 빈도수 확인
unique_elements, counts_elements = np.unique(labels, return_counts=True)
print("각 카테고리에 대한 빈도수:")
print(np.asarray((unique_elements, counts_elements)))

# 각 카테고리에 대한 빈도수:
# [['가슴통증' '결막염' '고열' '두통' '마른기침' '몸살' '미각이상' '설사' '손발가락변색' '운동장애' '인후통'
#   '피로감' '피부발진' '호흡곤란' '후각이상']
#  ['1205' '853' '2984' '914' '1659' '923' '448' '857' '151' '1002' '1081'
#   '2109' '791' '2239' '788']]


# %% 정수 인코딩, 패딩
word_index['OOV'] = 0
index_word[0] = 'OOV'

encoded=[]
for s in sentences:
    temp = []
    for w in s.split():
        try:
            temp.append(word_index[w])
        except KeyError:
            temp.append(word_index['OOV'])
    encoded.append(temp)


sents_len = 128 # 길이 500이상 그냥 자르기..
x_data = pad_sequences(encoded, maxlen=sents_len)

# y_data(category) 데이터 정수 인덱싱
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

category_set = list(set(labels))
category_sort = sorted(category_set)
category_to_idx = {word: index for index, word in enumerate(category_sort)}
idx_to_category = {index: word for index, word in enumerate(category_sort)}

y_data = []

for word in labels:
    y_data.append(category_to_idx[word])


from keras.utils import np_utils

y_data = np.asarray(y_data)
y_data = np_utils.to_categorical(y_data)

print(x_data.shape) # , 128
print(y_data.shape) # , 15


# %% 3
# 데이터셋 섞기
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, random_state=66, test_size=0.2)

print(X_train.shape)
print(X_test.shape)

# %% 4 Embedding Layer에 주입할 w2v 모델 처리
# embedding_index = {단어:[단어벡터], ...}

import gensim.models as g

# w2v_model = g.Doc2Vec.load('model/nin20200222_all_.model')
w2v_model = g.Doc2Vec.load('model/covid19_min2.model')

vocab = list(w2v_model.wv.vocab)
vector = w2v_model[vocab]

max_words = len(vocab)
embedding_dim = 256

print(max_words) # 7931


# %%
embedding_index = {}
for i in range(len(vocab)):
    embedding_index.setdefault(vocab[i], list(vector[i]))


# (max_words, embedding_dim) 크기인 임베딩 행렬을 임베딩 층에 주입.

embedding_matrix = np.zeros((max_words, embedding_dim))

for word, i in word_index.items():
    embedding_vector = embedding_index.get(word)
    if i < max_words:
        if embedding_vector is not None:
            # 임베딩 인덱스에 없는 단어는 0
            embedding_matrix[i] = embedding_vector
            # word_index의 index는 1부터 시작

# print(embedding_matrix[0])
print(len(embedding_matrix))


# %%