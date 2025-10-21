import requests, json

client_id = "gY8EgHoE5ke4rR8zXxC8"
client_secret = "FmAojzW2Ic"

url = "https://openapi.naver.com/v1/search/book.json"

headers = {"X-Naver-Client-Id": client_id, "X-Naver-Client-Secret": client_secret}
query_list = [
    "python 영진닷컴",
    "python 한빛미디어",
    "python 길벗",
    "python O'Reilly",
    "python packt",
    "python CRC",
]
params = {"query": "python"}
display = 100
start = 1

total_data = []
for query in query_list:
    try:
        while True:
            response = requests.get(
                url,
                headers=headers,
                params={"query": query, "display": display, "start": start},
            )
            if response.status_code == 200:

                data = response.json()
                items = data["items"]

                if not items:
                    break

                total_data.extend(items)
                start += 100
                if start > 1000:
                    break

            else:
                print("오류:", response.status_code)
                break
    except ValueError:
        print("JSON 데이터가 아닙니다")
    except requests.exceptions.RequestException as e:
        print("요청 중 오류 발생:", e)

if total_data:
    try:
        with open("book_search.json", "w", encoding="utf-8") as f:
            json.dump(total_data, f, indent=4, ensure_ascii=False)
            print(f"{len(total_data)}개의 데이터 저장 완료")
    except(OSEroor, PermissionError) as e:
        print(f:"")