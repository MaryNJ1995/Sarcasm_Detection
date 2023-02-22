import pandas as pd

df = pd.read_csv('testak.csv')
cnt = 0
for txt, tru, pred in zip(df.text, df.true, df.predict):
    # print(txt, tru, pred)
    if tru == 1 and pred == 1:
        cnt = cnt + 1
        print(txt)
print(cnt)
