import requests, json

url = "https://api.github.com"
response = requests.get(url)
print("상태 코드:", response.status_code)
print("응답 내용:", response.text[:200])
print("응답 내용:", response.json())
