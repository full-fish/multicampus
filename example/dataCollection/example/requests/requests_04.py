import requests

url = "https://httpbin.org/post"
payload = {"user": "kim", "id": 123}
response = requests.post(url, json=payload)
print(response.json())
