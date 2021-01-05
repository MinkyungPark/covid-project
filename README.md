# covid-project
covid19 symptom chatbot

# 전처리
### 0. 문장분리 kss
- cython 설치 후 pip install kss
- 사용법 <br>
> s = "회사 동료 분들과 다녀왔는데 분위기도 좋고 음식도 맛있었어요 다만, 강남 토끼정이 강남 쉑쉑버거 골목길로 쭉 올라가야 하는데 다들 쉑쉑버거의 유혹에 넘어갈 뻔 했답니다 강남역 맛집 토끼정의 외부 모습." <br>
> print('\n'.join(kss.split_sentences(s))) <br>
> for sent in kss.split_sentences(s): print(sent) <br>
결과 list안에 분리된 문장 담김. ['첫번째 줄','두번째줄','세번째줄']

<br>
<br>

### 1. Remove punctuation

<br>
<br>

### 2. Tokenize
- pip install git+https://github.com/haven-jeon/PyKoSpacing.git

<br>
<br>

### 3. Spelling Checking
- pip install git+https://github.com/ssut/py-hanspell
- 의존성 requests <br>
- 사용법 : https://github.com/ssut/py-hanspell <br>

> result = spell_checker.check(u'안녕 하세요. 저는 한국인 입니다. 이문장은 한글로 작성됬습니다.') <br>
**결과**<br>
> Checked(result=True, original='안녕 하세요. 저는 한국인 입니다. 이문장은 한글로 작성됬습니다.', checked='안녕하세요. 저는 한국인입니다. 이 문장은 한글로 작성됐습니다.', errors=4, words=OrderedDict([('안녕하세요.', 2), ('저는', 0), ('한국
인입니다.', 2), ('이', 2), ('문장은', 2), ('한글로', 0), ('작성됐습니다.', 1)]), time=0.3301119804382324)
<br><br>

> print(result.checked) <br> 
**결과** <brs>
안녕하세요. 저는 한국인입니다. 이 문장은 한글로 작성됐습니다.

<br>
<br>

### 4. Pos Tag + Stemming
- khaiii, mecab -> linux
- konlpy kkma <br>
significant_tags = ['NNG', 'NNP', 'NNB', 'VV', 'VA', 'VX', 'MAG', 'MAJ', 'XSV', 'XSA']
- Stemming : 동사를 원형으로 복원 <br>
<참고> https://lovit.github.io/nlp/2018/06/07/lemmatizer/ <br>
<규칙> <br>
1. NNG|NNP|NNB + XSV|XSA --> NNG|NNP|NNB + XSV|XSA + 다
2. NNG|NNP|NNB + XSA + VX --> NNG|NNP + XSA + 다s
3. VV --> VV + 다
4. VX --> VX + 다