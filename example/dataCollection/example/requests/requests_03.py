import requests

url = "https://httpbin.org/post"
d = {"name": "홍길동", "age": 30}
response = requests.post(url, data=d)
print(response.status_code)
print(response.text)
