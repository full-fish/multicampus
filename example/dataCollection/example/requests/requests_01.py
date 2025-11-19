import requests

url = "https://httpbin.org/json"
response = requests.get(url)
data = response.json()
print(type(data))  # <class 'dict'>
print(data["slideshow"]["title"])
