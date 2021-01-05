'''
공통적인 증상은 다음과 같습니다.
- 발열
- 마른기침
- 피로감

드물지만 다음과 같은 증상이 나타날 수도 있습니다.
- 몸살
- 인후통
- 설사
- 결막염
- 두통
- 미각 또는 후각 상실
- 피부 발진, 손가락 또는 발가락 변색

심각한 증상은 다음과 같습니다.
- 호흡 곤란 또는 숨 가쁨
- 가슴 통증 또는 압박감
- 언어 또는 운동 장애
'''


import time
import random
import re
import pandas as pd
from datetime import datetime
import numpy as np
import requests
import lxml.html


def check_number(text):
    retext = text.replace(",","")
    lnum = int(re.findall(r'\d+', retext)[1])
    rnum = int(re.findall(r'\d+', retext)[2])
    
    if lnum == rnum:
        return True
    else:
        return False
    
        
def naver_crawler(quest, year):
    query = quest
    period_from =  str(year) + '.03.01.'
    period_to =  str(year) + '.10.17.'
    
    naver_front = 'https://kin.naver.com/search/list.nhn?query='
    period_from = '&period=' + str(period_from)
    period_to = '%7c' + str(period_to)
    kin_url = naver_front + query + period_from + period_to + '&page={}'
    
    title = []
    ques = []
    answer = []
    date = []
    url = []
    
    page = 0
    while True:
        page += 1 
        time.sleep(random.uniform(1,4))
        res = requests.get(kin_url.format(page)) 
        time.sleep(random.uniform(1,4))
        root = lxml.html.fromstring(res.text)
        
        try:
            css = root.cssselect('.number em')
            if(check_number(css[0].text_content())):
                break
        except:
            print("crawl error: " , kin_url.format(page))
            page -= 1
            random.uniform(30,60)
            next
        if page > 150:
            break
        
        
        css = root.cssselect('.basic1 dt')
        links = []

        for i in css:
            try:
                if 'https:' in i.cssselect('a')[0].attrib['href'] :
                    links.append(i.cssselect('a')[0].attrib['href'])
                else:
                    next
            except IndexError:
                next
        
                        
        for link in links:
            res_links = requests.get(link)
            time.sleep(random.uniform(1,4))
            root_links = lxml.html.fromstring(res_links.text) 

            for br in root_links.xpath("*//p"):
                br.tail = "\n" + br.tail if br.tail else "\n"
            
            for br in root_links.xpath("*//br"):
                br.tail = "\n" + br.tail if br.tail else "\n"

            try:
                title.append(root_links.cssselect('.title')[0].text_content().strip())
            except IndexError:
                title.append('')
            try:
                ques.append(root_links.cssselect('.c-heading__content')[0].text_content().strip())
            except IndexError:
                ques.append('')
            try:
                answer.append(root_links.cssselect('._endContentsText')[0].text_content().strip())
            except IndexError:
                answer.append(np.NaN)
            try:
                date.append(root_links.cssselect('.c-userinfo__info')[0].text_content().strip()[3:])
            except IndexError:
                date.append(np.NaN)
            url.append(link)
        
    DF = pd.DataFrame({'Title':title, 'Question':ques,'Answer':answer,'Date':date, 'URL':url, 'Keyword':quest}) 
    return DF 

# lis = [['코로나 인가요']]


lis = [['실어증']]


def run(lis):
    for li in lis:
        df = pd.DataFrame({'Title':[0], 'Question':[0],'Answer':[0],'Date':[0], 'URL':[0], 'Keyword':[0]})

        keywords = li

        for k in keywords:
            for i in range(2019,2020): 
                df = pd.concat([df, naver_crawler(k ,i)], axis=0)        
                print(k, i, len(df))
                time.sleep(random.uniform(40,60))  #접속차단 방지
            
        print(df) 

        filename = "crawl_" + keywords[0] + ".csv"
        df = df.dropna()
        df = df.drop_duplicates('Question',keep='first')
        df.to_csv(filename, encoding = 'utf-8-sig')


if __name__ == "__main__":
    run(lis)