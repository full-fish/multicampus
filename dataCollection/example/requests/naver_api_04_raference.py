import requests
import json
import sys

# 네이버 API 인증 정보
client_id = "dnvKKBCqgwA7o2TyZjPm"
client_secret = "qP_srPPTRC"

headers = {"X-Naver-Client-Id": client_id, "X-Naver-Client-Secret": client_secret}

# 검색어 목록
query_list = [
    "python 영진닷컴",
    "python 한빛미디어",
    "python 길벗",
    "python O'Reilly",
    "python Packt",
    "python CRC",
]

url = "https://openapi.naver.com/v1/search/book.json"

# 결과 저장 리스트

total_cnt = 1
try:
    # 반복 수집
    for query in query_list:
        result = []
        display = 100  # 요청당 최대 100건
        start = 1
        total = None
        write_flag = False

        print(f"\n\n '{query}' 검색 시작...")

        while True:
            params = {"query": query, "display": 100, "start": start}

            response = requests.get(url, headers=headers, params=params)

            if response.status_code == 200:
                data = response.json()

                if total == None:
                    total = data.get("total", 0)
                    print(f"총 {total}건의 검색결과가 조회됨")
                    if total == 0:
                        print("검색 결과 데이터가 없음")
                        break

                items = data.get("items", [])
                curr_count = len(items)
                print(f"이번요청으로 {curr_count}건의 결과를 수집함")
                if curr_count == 0:
                    print("현재 요청에 대한 수집결과가 없음")
                    break

                for cnt, item in enumerate(items, total_cnt):
                    item["info_num"] = cnt
                    result.append(item)

                start += curr_count
                total_cnt += curr_count

                if start >= total or start >= 1000:
                    write_flg = True
                    break

            else:
                print("오류 발생:", response.status_code)

        if write_flg:
            print(f"총{start-1}건의 정보를 수집함")
            try:
                with open("all_ai_book_search.json", "a", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=4)
                print(f"{query} 검색결과 ai_book_search.json에 저장완료")
            except (OSError, PermissionError) as e:
                print(f"{query} 정보 파일 저장 중 오류:", e)


except requests.exceptions.RequestException as e:
    print("요청 중 오류 발생:", e)

except ValueError:
    print("JSON 응답이 아닙니다. 원본 텍스트 출력:")
    print(response.text[:300])

except Exception as e:
    # 가장 마지막에 잡히는 일반 예외 (예상 못한 경우)
    print("예기치 못한 오류:", type(e).__name__, "-", e)

else:  # 예외가 발생하지 않았을 경우에만 실행
    print(f"모든 검색어에 대한 데이터 수집 : {total_cnt-1} 건 완료")
