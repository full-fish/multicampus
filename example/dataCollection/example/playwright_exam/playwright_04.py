import asyncio
from playwright.async_api import async_playwright


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("https://quotes.toscrape.com")
        quotes = page.locator("div.quote")
        count = await quotes.count()
        for i in range(count):
            q = quotes.nth(i)
            text = await q.locator("span.text").text_content()
            author = await q.locator("small.author").text_content()
            print(f"{i+1}. {author}: {text}")
        await browser.close()


asyncio.run(main())
