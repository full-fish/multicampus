import asyncio
import json
import aiomysql  # 비동기 MySQL 라이브러리
from playwright.async_api import async_playwright, TimeoutError, Error

# 📘 대상 URL 목록
URLS = [
    "https://quotes.toscrape.com/page/1/",
    "https://quotes.toscrape.com/page/2/",
    "https://quotes.toscrape.com/page/3/",
]

# MySQL 연결 정보
DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",  # ← 본인 MySQL 계정
    "password": "1234",  # ← 본인 비밀번호
    "db": "testdb",  # 데이터베이스명
}


# ⚙️ 1️⃣ 테이블 생성 함수
async def init_db():
    pool = None
    try:
        async with aiomysql.connect(**DB_CONFIG) as conn:
            async with conn.cursor() as cur:
                # 데이터베이스 없으면 생성
                # await cur.execute("CREATE DATABASE IF NOT EXISTS testdb")
                # await cur.execute("USE testdb")

                # 테이블 생성
                create_table_sql = """
                CREATE TABLE IF NOT EXISTS quote_scrap (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    page_num INT,
                    author VARCHAR(255),
                    text TEXT
                );
                """
                await cur.execute(create_table_sql)
                await conn.commit()

        # DB_CONFIG['db']= "scrape"
        # 비동기 MySQL 연결 풀 생성
        pool = await aiomysql.create_pool(**DB_CONFIG, minsize=1, maxsize=5)
    except aiomysql.Error as e:
        print("데이터베이스 오류 :", e)
    except Exception as e:
        print("데이터베이스 일반오류eee :", e)
    finally:
        return pool


# 2️⃣ 페이지 크롤링 함수
async def scrape_page(page, url, pool):
    try:
        await page.goto(url)
        quotes = await page.query_selector_all(".quote")
        page_num = int(url.split("/")[-2])

        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                for q in quotes:
                    text_el = await q.query_selector("span.text")
                    author_el = await q.query_selector("small.author")
                    text = await text_el.inner_text()
                    author = await author_el.inner_text()

                    # MySQL에 삽입
                    insert_sql = """
                        INSERT INTO quote_scrap (page_num, author, text)
                        VALUES (%s, %s, %s);
                    """
                    await cur.execute(insert_sql, (page_num, author, text))
                await conn.commit()

        print(f"✅ {url} 저장 완료 ({len(quotes)}개)")

    except TimeoutError:
        print("대기 시간 초과 (TimeoutError)")
    except Error as e:
        print("Playwright 관련 오류:", e)
    except aiomysql.Error as e:
        print("데이터 베이스 오류:", e)
    except Exception as e:
        print("일반 Python 오류:", e)


# ⚙️ 3️⃣ 메인 함수
async def main():
    # DB 초기화
    pool = await init_db()
    if not pool:
        return

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()

            # 병렬 크롤링 실행
            tasks = []
            for url in URLS:
                page = await context.new_page()
                tasks.append(scrape_page(page, url, pool))

            await asyncio.gather(*tasks)

            await context.close()
            await browser.close()
            pool.close()
            await pool.wait_closed()

        print("🎉 모든 데이터가 MySQL에 저장되었습니다.")

    except TimeoutError:
        print("대기 시간 초과 (TimeoutError)")
    except Error as e:
        print("Playwright 관련 오류:", e)
    except Exception as e:
        print("일반 Python 오류:", e)


# 4️⃣ 실행
if __name__ == "__main__":
    asyncio.run(main())
