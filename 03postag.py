'''
kkma.morphs         #형태소 분석
kkma.nouns          #명사 분석
kkma.pos            #형태소 분석 태깅
kkma.sentences      #문장 분석
 
# 사용예시
print(kkma.morphs(u'공부를 하면할수록 모르는게 많다는 것을 알게 됩니다.'))
['공부', '를', '하', '면', '하', 'ㄹ수록', '모르', '는', '것', '이', '많', '다는', '것', '을', '알', '게', '되', 'ㅂ니다', '.']
 
print(kkma.nouns(u'대학에서 DB, 통계학, 이산수학 등을 배웠지만...'))
['대학', '통계학', '이산', '이산수학', '수학', '등']
 
print(kkma.pos(u'다 까먹어버렸네요?ㅋㅋ'))
[('다', 'MAG'), ('까먹', 'VV'), ('어', 'ECD'), ('버리', 'VXV'), ('었', 'EPT'), ('네요', 'EFN'), ('?', 'SF'), ('ㅋㅋ', 'EMO')]
 
print(kkma.sentences(u'그래도 계속 공부합니다. 재밌으니까!'))
['그래도 계속 공부합니다.', '재밌으니까!']

'''
import pandas as pd
import re as re
from konlpy.tag import Kkma


def pos_tag(sentences, labels):
    kkma = Kkma()
    significant_tags = ['NNG', 'NNP', 'NNB', 'VV', 'VA', 'VX', 'MAG', 'MAJ', 'XSV', 'XSA']
    # 일반 명사, 고유 명사, 의존 명사, 동사, 형용사, 보조 용언, 일반 부사, 접속 부사, 동사 파생 접미사, 형용사 파생 접미사

    s, l = [], []
    for sent,label in zip(sentences, labels):
        tmp = []
        for word, tag in kkma.pos(sent):
            print(word+tag+' ')
            if tag in significant_tags:
                tmp.append(word+'/'+tag)
        s.append(stemming_text(tmp))
        l.append(label)

    result = pd.DataFrame([x for x in zip(s,l)], columns=['sentences','labels'])

    return result


def stemming_text(text):
    p1 = re.compile('[가-힣A-Za-z0-9]+/NN. [가-힣A-Za-z0-9]+/XS.')
    p2 = re.compile('[가-힣A-Za-z0-9]+/NN. [가-힣A-Za-z0-9]+/XSA [가-힣A-Za-z0-9]+/VX')
    p3 = re.compile('[가-힣A-Za-z0-9]+/VV')
    p4 = re.compile('[가-힣A-Za-z0-9]+/VX')

    corpus = []
    for sent in text:
        ori_sent = sent
        mached_terms = re.findall(p1, ori_sent)
        for terms in mached_terms:
            ori_terms = terms
            modi_terms = ''
            for term in terms.split(' '):
                lemma = term.split('/')[0]
                tag = term.split('/')[-1]
                modi_terms += lemma
            modi_terms += '다/VV'
            ori_sent = ori_sent.replace(ori_terms, modi_terms)
        
        mached_terms = re.findall(p2, ori_sent)
        for terms in mached_terms:
            ori_terms = terms
            modi_terms = ''
            for term in terms.split(' '):
                lemma = term.split('/')[0]
                tag = term.split('/')[-1]
                if tag != 'VX':
                    modi_terms += lemma
            modi_terms += '다/VV'
            ori_sent = ori_sent.replace(ori_terms, modi_terms)

        mached_terms = re.findall(p3, ori_sent)
        for terms in mached_terms:
            ori_terms = terms
            modi_terms = ''
            for term in terms.split(' '):
                lemma = term.split('/')[0]
                tag = term.split('/')[-1]
                modi_terms += lemma
            if '다' != modi_terms[-1]:
                modi_terms += '다'
            modi_terms += '/VV'
            ori_sent = ori_sent.replace(ori_terms, modi_terms)

        mached_terms = re.findall(p4, ori_sent)
        for terms in mached_terms:
            ori_terms = terms
            modi_terms = ''
            for term in terms.split(' '):
                lemma = term.split('/')[0]
                tag = term.split('/')[-1]
                modi_terms += lemma
            if '다' != modi_terms[-1]:
                modi_terms += '다'
            modi_terms += '/VV'
            ori_sent = ori_sent.replace(ori_terms, modi_terms)
        corpus.append(ori_sent)

    corpus = remove_stopword_text(corpus)

    # corpus = ['코로나/NNG', '걸리다/VV', '가슴/NNG', '통증/NNG', '있다/VV', '하다/VV', '걸리다/VV', '식/NNB', '아프/VA']
    # String으로 품사를 제거해서 리턴
    result = ""
    for word in corpus:
        result += word.split('/')[0] + ' '

    return result


# 불용어 삭제
def remove_stopword_text(text):
    stopwords = ['데/NNB', '좀/MAG', '수/NNB', '등/NNB']

    corpus = []
    for sent in text:
        modi_sent = []
        for word in sent.split(' '):
            if word not in stopwords:
                modi_sent.append(word)
        corpus.append(' '.join(modi_sent))
    return corpus


if __name__ == '__main__':
    q = pd.read_csv('data/spellchecked.csv', engine='python', encoding='utf-8', names=['sentences','labels'])[1:].dropna()

    sentences = q.sentences
    labels = q.labels

    pos_tag(sentences, labels).to_csv('data/postagged.csv', index=False, encoding='utf-8-sig')


