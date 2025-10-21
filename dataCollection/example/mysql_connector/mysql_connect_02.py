import requests, json

import mysql.connector
from mysql.connector import Error


# api 조회 단
def get_book_data():
    client_id = "gY8EgHoE5ke4rR8zXxC8"
    client_secret = "FmAojzW2Ic"

    url = "https://openapi.naver.com/v1/search/book.json"

    headers = {"X-Naver-Client-Id": client_id, "X-Naver-Client-Secret": client_secret}

    query = "인공지능"
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
    print(f"{query}에 대해서 책 조회. {len(total_data)}개의 데이터 조회")
    return total_data


# sql 단
def sql(row_data):
    try:
        # # 데이터베이스 연결
        conn = mysql.connector.connect(
            host="localhost",  # 또는 IP 주소
            user="root",  # MySQL 사용자
            password="sktkfka5",
            database="testdb",  # 사용할 데이터베이스명
        )
        if conn.is_connected():
            print("mysql 연결성공")

        # 커서 생성
        cursor = conn.cursor()

        # ? 생성
        cursor.execute(
            """
                    create table if not exists books(
                    id int auto_increment primary key,
                    title text not null,
                    originallink text,
                    link text not null,
                    description text not null,
                    pubdate text not null
                    )
        """
        )
        sql = "insert into books (title, originallink, link, description, pubdate) values (%s, %s, %s, %s, %s)"
        val_list = [
            (
                data.get("title"),
                data.get("originallink"),
                data.get("link"),
                data.get("description"),
                data.get("pubdate"),
            )
            for data in row_data
        ]

        cursor.executemany(sql, val_list)
        conn.commit()
        print("데이터 저장 insert")

        # ? 조회
        # cursor.execute("SELECT * FROM books")
        # # 전체 조회
        # rows = cursor.fetchall()
        # print(rows)
        # # 한 행씩
        # while True:
        #     row = cursor.fetchone()
        #     if row == None:
        #         break
        #     print(row)

        # ? 수정
        # cursor.execute('update users set age = 26 where name = "김승현"')
        # conn.commit()

        # ? 삭제
        # cursor.execute('delete from users where name="김승현"')
        # conn.commit()
    except Error as err:
        print("데이터 베이스 오류", err)
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


sql(get_book_data())
