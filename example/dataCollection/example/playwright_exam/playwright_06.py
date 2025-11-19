import asyncio, json
from playwright.async_api import async_playwright

url = "https://quotes.toscrape.com"
num = 3


async def main(num):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        page_arr = [await browser.new_page() for _ in range(num)]

        await asyncio.gather(
            *[get_data_page(page, i + 1) for i, page in enumerate(page_arr)]
        )
        await browser.close()


async def get_data_page(page, num):
    await page.goto(f"{url}/page/{num}")
    quotes = page.locator("div.quote")
    count = await quotes.count()
    data = [f"-----{num}번째 페이지 데이터----"]
    for i in range(count):
        q = quotes.nth(i)
        text = await q.locator("span.text").text_content()
        author = await q.locator("small.author").text_content()
        data.append({"author": author, "text": text})
    with open(f"{num}.page data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


asyncio.run(main(num))
