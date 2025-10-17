from bs4 import BeautifulSoup

html_doc = """
<html>
  <head>
    <title>My Website</title>
  </head>
  <body>
    <h1>안녕하세요!</h1>
    <p>이건 <a href="https://example.com">예제 링크</a>입니다.</p>
  </body>
</html>
"""
soup = BeautifulSoup(html_doc, "html.parser")

print("Title:", soup.title.string)
print("Heading:", soup.h1.text)
# 이런식으로 find해도 똑같음
# print("Title:",soup.find('title').string)
# print("Heading:", soup.find('h1').text)
print("Link:", soup.a["href"])
