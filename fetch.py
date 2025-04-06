import requests
from bs4 import BeautifulSoup

url="https://www.occamsadvisory.com/"
response=requests.get(url)
soup=BeautifulSoup(response.text,"html.parser")
text=soup.get_text()

with open("occams_contant.txt","w",encoding="utf-8")as file:
    file.write(text)

print("saved in text")