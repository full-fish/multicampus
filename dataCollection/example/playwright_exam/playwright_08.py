import asyncio, json
from playwright.async_api import async_playwright
import aiomysql

url = "https://quotes.toscrape.com"
num = 3


async def main(num):
    pool = await aiomysql.create_pool(
        host="localhost", port=3306, user="root", password="sktkfka5", db="testdb"
    )

    try:
        await create_table(pool)

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page_arr = [await browser.new_page() for _ in range(num)]

            all_data = await asyncio.gather(
                *[get_data_page(page, i + 1) for i, page in enumerate(page_arr)]
            )
            print("all_data개수", len(all_data))
            tasks = [
                insert_data(pool, page_data) for page_data in all_data if page_data
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

            await browser.close()
    finally:
        pool.close()
        await pool.wait_closed()


async def get_data_page(page, num):
    # await page.goto(f"{url}/page/{num}")
    # quotes = page.locator("div.quote")
    # count = await quotes.count()
    # data = []

    # print(f"-----{num}번째 페이지 데이터---- 개수: {count}")
    # for i in range(count):
    #     q = quotes.nth(i)
    #     text = await q.locator("span.text").text_content()
    #     author = await q.locator("small.author").text_content()
    #     data.append({"author": author, "text": text})
    
    #? 작동 방식이 다름 
    await page.goto(f"{url}/page/{num}")
    data = []

    for e in await page.locator("div.quote").all():
        text = await e.locator("span.text").text_content()
        author = await e.locator("small.author").text_content()
        data.append({"author": author, "text": text})
    return data


async def create_table(pool):
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                CREATE TABLE IF NOT EXISTS quotes (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    author VARCHAR(255),
                    text TEXT
                )
                """
            )
        await conn.commit()


async def insert_data(pool, quotes_data):
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            values = [(quote["author"], quote["text"]) for quote in quotes_data]
            await cur.executemany(
                "INSERT INTO quotes (author, text) VALUES (%s, %s)", values
            )
        await conn.commit()
    print(f"{len(quotes_data)}개의 명언이 데이터베이스에 저장되었습니다.")


asyncio.run(main(num))
