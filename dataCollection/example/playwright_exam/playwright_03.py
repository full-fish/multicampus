import asyncio
from playwright.async_api import async_playwright


async def main():
    async with async_playwright() as p:

        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()

        await page.goto("https://quotes.toscrape.com/")
        await page.click("a[href='/login']")
        await page.fill("input#username", "admin")
        await page.fill("input#password", "admin")

        await page.click("input[type='submit']")
        await page.wait_for_selector("div.quote")
        html = await page.content()
        if "Logout" in html:
            print("로그인 성공!")
        else:
            print("로그인 실패!")
        await asyncio.sleep(9999)
        await browser.close()


asyncio.run(main())
