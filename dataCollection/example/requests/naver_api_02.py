import requests, json

client_id = "gY8EgHoE5ke4rR8zXxC8"
client_secret = "FmAojzW2Ic"

url = "https://openapi.naver.com/v1/search/book.json"

params = {"query": "python"}

headers = {"X-Naver-Client-Id": client_id, "X-Naver-Client-Secret": client_secret}

response = requests.get(url, headers=headers, params=params)
# response = requests.get(url + "?query=phthon", headers=headers)

if response.status_code == 200:
    try:
        data = response.json()

        with open("book_serach.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print("저장 완")

    except ValueError:
        print("JSON 데이터가 아닙니다")


else:
    print("오류:", response.status_code)
