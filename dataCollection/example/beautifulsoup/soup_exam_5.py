import requests
from bs4 import BeautifulSoup

url = "https://quotes.toscrape.com/"

response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

quotes = soup.select("div.quote")

# print(soup.find("span", class_="text").text)
# print(quotes[0].find("span", class_="text").text)

# print("----------------------")

# for i in quotes[:3]:
#     print(i.find("span", class_="text").text)

# print("----------------------")

# for i in quotes:
#     author = i.select_one("small.author").text
#     tags = [t.text for t in i.select("div.tags>a.tag")]
#     print(f"{author}: {tags}")

# for i in soup.find_all("div", class_="quote"):
#     author = i.find("small", class_="author").text
#     tags = [t.text for t in i.find_all("a", class_="tag")]
#     print(f"{author}: {tags}")

https://httpbin.org/json
authors = {}
for i in quotes:
    quote = i.select_one(".text").text
    author = i.select_one("small.author").text
    if author != "Albert Einstein":
        continue
    tags = [t.text for t in i.select("div.tags>a.tag")]
    if author not in authors:
        authors[author] = {"quote_arr": [], "tag_arr": []}
        authors[author]["quote_arr"].append(quote)
        authors[author]["tag_arr"].append(tags)

for author, data in authors.items():
    print(f"==={author}의 인용구===")
    for quote in data["quote_arr"]:
        print(quote)
    print(f"==={author}의 인용구별 태그===")
    for tag in data["tag_arr"]:
        print(tag)
    print("\n------------\n")

# print(f"===[{author}]의 인용구===")
# for n in quote_arr:
# print(n.text)
