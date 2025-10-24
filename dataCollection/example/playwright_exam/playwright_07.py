import asyncio
from playwright.async_api import async_playwright


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("http://3.39.100.13")
        await page.locator('input[placeholder="Username"]').fill("aaa")
        await page.locator('input[placeholder="Password"]').fill("111")
        await page.locator('button[name="login"]').click()
        await page.locator('button:has-text("ADMIN")').click()
        await page.wait_for_selector("h1")
        print(await page.locator("h1").text_content())
        print(await page.locator("h1").inner_text())
        print(await page.locator("h1").inner_html())

        await browser.close()


asyncio.run(main())
