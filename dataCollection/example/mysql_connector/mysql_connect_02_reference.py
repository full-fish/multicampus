import requests
import json
import mysql.connector
from mysql.connector import Error

# ============================================
# 네이버 뉴스 API 설정
# ============================================
url = "https://openapi.naver.com/v1/search/book.json"
query = "인공지능"

headers = {
    "X-Naver-Client-Id": "dnvKKBCqgwA7o2TyZjPm",
    "X-Naver-Client-Secret": "qP_srPPTRC",
}

# ============================================
# MySQL 연결 (예외처리 포함)
# ============================================
conn = None
cursor = None
try:
    conn = mysql.connector.connect(
        host="localhost", user="root", password="1234", database="testdb"
    )

    if conn.is_connected():
        print("MySQL 연결 성공!")
        cursor = conn.cursor()

        # -------------------------------
        # 테이블 생성 (없으면 자동 생성)
        # -------------------------------
        create_table_query = """
        CREATE TABLE IF NOT EXISTS bookdata (
            id INT AUTO_INCREMENT PRIMARY KEY,
            title TEXT,
            originallink TEXT,
            link TEXT,
            description TEXT,
            pubDate VARCHAR(50)
        )
        """
        cursor.execute(create_table_query)
        conn.commit()
        print("bookdata 테이블 준비 완료")

except Error as e:
    print("MySQL 연결 실패:", e)
    exit()  # DB 연결 실패 시 프로그램 종료

# ============================================
# 뉴스 데이터 수집 (requests 예외처리)
# ============================================
result = []
start = 1
total = None

try:
    while True:
        params = {"query": query, "display": 100, "start": start}
        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            data = response.json()

            if total == None:
                total = data.get("total", 0)  # data['total']
                if total == 0:
                    break

            items = data.get("items", [])
            curr_cnt = len(items)
            if curr_cnt == 0:
                break

            result.extend(items)
            start += curr_cnt

            if start >= total or start >= 1000:
                write_flag = True
                break
        else:
            print("오류 발생 :", response.status_code)

except requests.exceptions.RequestException as e:
    print("요청 중 오류 발생:", e)
except ValueError:
    print("응답이 JSON 데이터가 아닙니다.")
    print(response.text[:200])
except Exception as e:
    print("오류:", e)
else:
    print(f"총 {len(result)}건의 책정보 수집 완료")

if len(result) == 0:
    print("수집내용이 없습니다.")
    if cursor:
        cursor.close()
    if conn and conn.is_connected():
        conn.close()
    print("연결 종료 완료")
    exit()


# ============================================
# MySQL 저장 (SQL 예외처리)
# ============================================
insert_sql = """
INSERT INTO bookdata (title, link, description, pubDate)
VALUES (%s, %s, %s, %s)
"""

cnt = 0
try:
    for item in result:
        values = (
            item.get("title", ""),
            item.get("link", ""),
            item.get("description", ""),
            item.get("pubDate", ""),
        )
        cursor.execute(insert_sql, values)
        cnt += 1

    conn.commit()
    print(f"{cnt}건의 데이터 저장 완료")

except Error as e:
    print(f"데이터 삽입 오류: {e}")

# ============================================
# 연결 종료
# ============================================
if cursor:
    cursor.close()
if conn and conn.is_connected():
    conn.close()
    print("연결 종료 완료")
