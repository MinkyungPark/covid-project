'''
문장 분리 후, clean text
'''

import pandas as pd
import re as re
import kss


file_name = ['가슴통증','결막염','고열','두통','마른기침','몸살','미각이상','설사','손발가락변색', \
    '운동장애','인후통','피로감','피부발진','호흡곤란','후각이상'] # 언어장애


def clean_text(text):
    # Common
    text = re.sub("(?s)<ref>.+?</ref>", "", text) # remove reference links
    text = re.sub("(?s)<[^>]+>", "", text) # remove html tags
    text = re.sub("&[a-z]+;", "", text) # remove html entities
    text = re.sub("(?s){{.+?}}", "", text) # remove markup tags
    text = re.sub("(?s){.+?}", "", text) # remove markup tags
    text = re.sub("(?s)\[\[([^]]+\|)", "", text) # remove link target strings
    text = re.sub("(?s)\[\[([^]]+\:.+?]])", "", text) # remove media links
    
    text = re.sub("[']{5}", "", text) # remove italic+bold symbols
    text = re.sub("[']{3}", "", text) # remove bold symbols
    text = re.sub("[']{2}", "", text) # remove italic symbols

    text = re.sub(r'[<>^?!.,ㅠㅜ@%`\\*=()/~#&\+á\xc3\xa1\-\|\:\;\-\_\~\$\'\"]', '',str(text)) #remove punctuation
    
    # text = re.sub(u"[^ \r\n{Hangul}.?!]", " ", text) # Replace unacceptable characters with a space.
    
    text = re.sub("[ ]{2,}", " ", text) # Squeeze spaces.
    
    return text


punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }

def clean_punc(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])
    
    for p in punct:
        text = text.replace(p, f' {p} ')
    
    specials = {'\u200b': ' ', '.': ' ', ',': ' ', '\ufeff': '', 'करना': '', 'है': ''}
    for s in specials:
        text = text.replace(s, specials[s])
    
    return text.strip()


def makeLabel(question, label):
    ques, lab = [], []
    for q in question: # type(q) -> str
        tmp = clean_text('\n'.join(kss.split_sentences(q)))
        sents = clean_punc(tmp, punct, punct_mapping)
        if sents is not None:
            ques.append(sents)
            lab.append(label)

    ques_col = pd.DataFrame(ques, columns=['sentences'])
    lab_col = pd.DataFrame(lab, columns=['label'])
    result = pd.concat([ques_col, lab_col], axis=1)

    return result


if __name__ == '__main__':
    for f in file_name:
        q = pd.read_csv('data/raw/' + f +'.csv', engine='python', encoding='utf-8-sig')
        q = q.dropna()

        question = q.Question
        label = f

        result = makeLabel(question, label)

        result[1:].to_csv('data/' + f + '.csv', index=False, encoding='utf-8-sig')