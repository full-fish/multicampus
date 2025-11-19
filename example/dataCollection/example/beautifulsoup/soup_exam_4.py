import requests
from bs4 import BeautifulSoup

url = "https://quotes.toscrape.com/"

response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# print(soup.footer.div.find_all("p"))
# print(soup.div.div.attrs)

for e in soup.find_all(attrs={"class": "text-muted"}):
    print(e)
for e in soup.find_all(class_="text-muted"):
    print(e)
