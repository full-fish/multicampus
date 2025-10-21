import requests
import json
import mysql.connector
from mysql.connector import Error


# ============================================
# DB 연결 함수
# ============================================
def connect_db(host, user, password, database):
    """MySQL 데이터베이스 연결"""
    try:
        conn = mysql.connector.connect(
            host="localhost",  # 또는 IP 주소
            user="root",  # MySQL 사용자
            password="sktkfka5",
            database="testdb",  # 사용할 데이터베이스명
        )
        if conn.is_connected():
            print("MySQL 연결 성공!")
            return conn
    except Error as e:
        print("MySQL 연결 실패:", e)
        return None


# ============================================
# 테이블 생성 함수
# ============================================
def create_table(conn):
    """책 정보 저장용 테이블 생성"""
    try:
        cursor = conn.cursor()
        sql = """
        CREATE TABLE IF NOT EXISTS bookdata (
            id INT AUTO_INCREMENT PRIMARY KEY,
            title TEXT,
            originallink TEXT,
            link TEXT,
            description TEXT,
            pubDate VARCHAR(50)
        )
        """
        cursor.execute(sql)
        conn.commit()
        cursor.close()
        print("bookdata 테이블 준비 완료")
    except Error as e:
        print("테이블 생성 실패:", e)


# ============================================
# 네이버 책검색 데이터 요청 함수
# ============================================
def fetch_naver_book(query, client_id, client_secret, limit=1000):
    """네이버 책 검색 API로 데이터 수집"""
    url = "https://openapi.naver.com/v1/search/book.json"
    headers = {"X-Naver-Client-Id": client_id, "X-Naver-Client-Secret": client_secret}

    result = []
    start = 1
    total = None
    try:
        while True:
            write_flag = False
            params = {"query": query, "display": 100, "start": start}
            response = requests.get(url, headers=headers, params=params)

            if response.status_code != 200:
                print(f"API 요청 실패 (status={response.status_code})")
                break

            data = response.json()

            if total is None:
                total = data.get("total", 0)
                if total == 0:
                    print("검색 결과 없음.")
                    break

            items = data.get("items", [])
            if not items:
                break

            result.extend(items)
            start += len(items)

            print(f"현재 {len(result)}건 수집 중...")

            if start >= total or start >= limit:
                write_flag = True
                break

    except requests.exceptions.RequestException as e:
        print("네트워크 오류:", e)
    except json.JSONDecodeError:
        print("JSON 파싱 오류")
    except Exception as e:
        print("오류:", e)

    else:
        if write_flag:
            print(f"총 {len(result)}건의 책정보 수집 완료")

    return result


# ============================================
# 데이터 정제 및 저장 함수
# ============================================
def save_to_mysql(conn, data):
    """수집된 책 정보를 MySQL에 저장"""
    insert_sql = """
    INSERT INTO bookdata (title, originallink, link, description, pubDate)
    VALUES (%s, %s, %s, %s, %s)
    """
    # cursor = None
    try:
        cursor = conn.cursor()
        cnt = 0
        for item in data:
            values = (
                item.get("title", ""),
                item.get("originallink", ""),
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
    finally:
        if cursor:
            cursor.close()


# ============================================
# 메인 실행 함수
# ============================================
def main():
    """전체 실행 흐름 제어"""
    # 네이버 API 설정
    CLIENT_ID = "dnvKKBCqgwA7o2TyZjPm"
    CLIENT_SECRET = "qP_srPPTRC"
    QUERY = "인공지능"

    # 1. DB 연결
    conn = connect_db("localhost", "root", "1234", "testdb")
    if conn is None:
        return

    # 2. 테이블 준비
    create_table(conn)

    # 3. 데이터 수집
    news_data = fetch_naver_book(QUERY, CLIENT_ID, CLIENT_SECRET)

    # 4. DB 저장
    if len(news_data) != 0:
        save_to_mysql(conn, news_data)
    else:
        print("저장할 데이터가 없습니다.")

    # 5. 연결 종료
    conn.close()
    print("연결 종료 완료")


# ============================================
# 실행
# ============================================
if __name__ == "__main__":
    main()
