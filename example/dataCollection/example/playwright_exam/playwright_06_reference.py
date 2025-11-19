import asyncio, json
from playwright.async_api import async_playwright

URLS = [
    "https://quotes.toscrape.com/page/1/",
    "https://quotes.toscrape.com/page/2/",
    "https://quotes.toscrape.com/page/3/"
]

async def scrape_page(page, url):
    await page.goto(url)
    quotes = await page.query_selector_all(".quote")
    result=[]
    for q in quotes:
        text_element = await q.query_selector("span.text")
        author_element = await q.query_selector("small.author")
        text = await text_element.inner_text()
        author = await author_element.inner_text()
        result.append({"text":text, "author":author})
    
    # print(await text.inner_text(), "-", await author.inner_text())
    
    with open(f"parallel_task_page{url[-2]}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
        

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        # context = await browser.new_context() #서로 다른 쿠키/스토리지 공간을 갖는 새 세션

        # 병렬 크롤링
        tasks = []
        for url in URLS:
            page = await browser.new_page()
            tasks.append(scrape_page(page, url))

        await asyncio.gather(*tasks)
        await browser.close()

asyncio.run(main())