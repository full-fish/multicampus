import requests, json

url = "https://www.python.org"
response = requests.get(url)
try:
    data = response.json()
except ValueError:
    print("JSON 응답이 아닙니다. 원본 텍스트 출력:")
    print(response.text[:300])
