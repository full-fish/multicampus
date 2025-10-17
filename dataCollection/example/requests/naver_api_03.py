import requests, json

client_id = "gY8EgHoE5ke4rR8zXxC8"
client_secret = "FmAojzW2Ic"

url = "https://openapi.naver.com/v1/search/book.json"


headers = {"X-Naver-Client-Id": client_id, "X-Naver-Client-Secret": client_secret}

params = {"query": "python"}
query = "python"
display = 100
start = 1

total_data = []
while True:
    response = requests.get(
        url,
        headers=headers,
        params={"query": query, "display": display, "start": start},
    )
    if response.status_code == 200:
        try:
            data = response.json()
            items = data["items"]

            if not items:
                break

            total_data.extend(items)
            start += 100
            if start > 1000:
                break

        except ValueError:
            print("JSON 데이터가 아닙니다")

    else:
        print("오류:", response.status_code)
        break
if total_data:
    with open("book_search.json", "w", encoding="utf-8") as f:
        json.dump(total_data, f, indent=4, ensure_ascii=False)
        print(f"{len(total_data)}개의 데이터 저장 완료")
