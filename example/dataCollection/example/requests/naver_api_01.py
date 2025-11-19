import requests, json

client_id = "gY8EgHoE5ke4rR8zXxC8"
client_secret = "FmAojzW2Ic"

url = "https://openapi.naver.com/v1/search/news.json"

params = {"query": "인공지능", "display": 3}

headers = {"X-Naver-Client-Id": client_id, "X-Naver-Client-Secret": client_secret}

response = requests.get(url, headers=headers, params=params)

if response.status_code == 200:
    data = response.json()
    # print(json.dumps(data, indent=4, ensure_ascii=False))
    for item in data["items"]:
        print(item["title"], "->", item["link"])


else:
    print("오류:", response.status_code)
