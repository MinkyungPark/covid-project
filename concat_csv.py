import pandas as pd


file_name = ['가슴통증','결막염','고열','두통','마른기침','몸살','미각이상','설사','손발가락변색', \
    '운동장애','인후통','피로감','피부발진','호흡곤란','후각이상'] # 언어장애

pds = []
for f in file_name:
    tmp = pd.read_csv('data/'+ f + '.csv', engine='python', encoding='utf-8-sig')
    pds.append(tmp)

pd.concat(pds, axis=0).to_csv('total.csv', index=False, encoding='utf-8-sig')