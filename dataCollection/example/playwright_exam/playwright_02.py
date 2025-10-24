import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup


async def crawl_quotes():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("https://quotes.toscrape.com/")
        html = await page.content()
        soup = BeautifulSoup(html, "html.parser")
    for q in soup.select("div.quote"):
        text = q.select_one("span.text").get_text(strip=True)
        author = q.select_one("small.author").get_text(strip=True)
        print(f"{author}: {text}")
    await browser.close()


asyncio.run(crawl_quotes())
