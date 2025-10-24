import asyncio
from playwright.async_api import async_playwright


async def auto_scroll(page):
    previous_height = 0
    while True:
        current_height = await page.evaluate("document.body.scrollHeight")
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await page.wait_for_timeout(1000)  # 로딩 대기
        new_height = await page.evaluate("document.body.scrollHeight")
        print(current_height)

        if new_height == previous_height:
            break
        previous_height = new_height


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("https://quotes.toscrape.com/scroll")

        await auto_scroll(page)

        quotes = await page.query_selector_all(".quote")
        print(f"총 {len(quotes)}개 항목 수집 완료")
        for q in quotes[:5]:  # 앞부분 5개만 출력
            text = await (await q.query_selector("span.text")).inner_text()
            author = await (await q.query_selector("small.author")).inner_text()
            print(f"{author}: {text}")
        await asyncio.sleep(9999)
        await browser.close()


asyncio.run(main())
