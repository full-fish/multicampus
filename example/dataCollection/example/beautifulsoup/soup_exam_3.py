import requests
from bs4 import BeautifulSoup

url = "https://quotes.toscrape.com/"

response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

quotes = soup.select("div.quote")
# print(f"총 {len(quotes)}개의 인용구가 있습니다.")

# for i, q in enumerate(quotes[:3], 1):
#     print(f"{i}: {q.select_one('span.text').text}")

print("\n저자별 태그 예시")
for q in quotes[:1]:
    print(q)
    author = q.select_one("small.author").text
    tags = [t.text for t in q.select("div.tags>a.tag")]
    print("-------------")
    print(tags)
