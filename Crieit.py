from bs4 import BeautifulSoup
import requests
import pandas as pd 

columns = ['title']
df = pd.DataFrame(columns=columns) 

base_url = "https://crieit.net/posts?page="

num = 1
while num <= 1:
 url=base_url+ str(num)
 r = requests.get(url)
 soup = BeautifulSoup(r.text, features="lxml")
 titles =soup.select("h5 a")
 if len(titles) == 0:
  print("これ以上記事はありません")  
  break
 num += 1
 for title in titles:
  se= pd.Series(title.text, columns)
  print(se)
  print("test")
  df = df.append(se, ignore_index=True)
  print(df)
  print("hello")

dhead = df.head()
print(dhead)