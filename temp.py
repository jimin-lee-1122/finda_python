print("Hello world!")

# 주석
"""
주석입니다

"""

# AAPL 주가를 가져오는 코드 
import requests
from bs4 import BeautifulSoup       # BeautifulSoup 모듈을 불러옵니다.  
url = "https://finance.naver.com/item/main.nhn?code=005930"
response = requests.get(url)       # url에 해당하는 웹페이지를 가져옵니다.
html = response.text               # html 소스를 가져옵니다.            
soup = BeautifulSoup(html, "html.parser") # BeautifulSoup를 이용해 html을 파싱합니다.
tags = soup.select("#_nowVal")     # id가 _nowVal인 태그를 가져옵니다.      
tag = tags[0]                      # 리스트의 첫번째 요소를 가져옵니다.
print(tag.text)                    # 태그의 텍스트를 출력합니다.    # 1 

