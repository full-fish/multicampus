import asyncio
from typing import List, Dict, Optional
from playwright.async_api import async_playwright

CHATGPT_URL = "https://chat.openai.com"  # 또는 "https://chatgpt.com"
USER_DATA_DIR = "./playwright-chatgpt-profile"  # 로그인 세션 보존용 디렉터리

# 안전장치 옵션
DRY_RUN = True  # True면 목록만 출력하고 실제 삭제는 안 함
MAX_TO_DELETE = None  # 정수로 제한하고 싶으면 예: 100
PAGE_SIZE = 50  # 한 번에 가져올 대화 수
SLEEP_BETWEEN_DELETES = 0.4  # 삭제 사이 딜레이(초)


async def fetch_conversations(page, limit=50) -> List[Dict]:
    conversations = []
    cursor: Optional[str] = None

    while True:
        url = f"/backend-api/conversations?limit={limit}"
        if cursor:
            url += f"&cursor={cursor}"

        batch = await page.evaluate(
            """async (url) => {
                const res = await fetch(url, { method: 'GET' });
                if (!res.ok) throw new Error('Failed to list conversations: ' + res.status);
                return res.json();
            }""",
            url,
        )

        items = batch.get("items", [])
        conversations.extend(items)

        cursor = batch.get("next_cursor")
        if not cursor or len(items) == 0:
            break

    return conversations


async def delete_conversation(page, conv_id: str) -> bool:
    return await page.evaluate(
        """async (convId) => {
            const res = await fetch(`/backend-api/conversation/${convId}`, {
                method: 'DELETE'
            });
            return res.ok;
        }""",
        conv_id,
    )


async def ensure_logged_in(page):
    await page.goto(CHATGPT_URL, wait_until="domcontentloaded")
    # 사이드바가 보이는지 등 간단한 로그인 판별 로직
    # 계정에 따라 초기 랜딩이 다를 수 있으니, 홈으로 한 번 더 이동
    await page.goto(f"{CHATGPT_URL}/", wait_until="domcontentloaded")

    # 토큰 유무로도 판별 가능하지만, 여기서는 대화 목록 API를 가볍게 호출해봄
    try:
        _ = await page.evaluate(
            """async () => {
                const res = await fetch('/backend-api/conversations?limit=1');
                return res.status;
            }"""
        )
    except Exception:
        pass


async def main():
    async with async_playwright() as p:
        # 영속 컨텍스트로 실행해서 한 번 로그인하면 계속 유지
        browser = await p.chromium.launch_persistent_context(
            USER_DATA_DIR,
            headless=False,  # 첫 로그인 편의를 위해 보이게
        )
        page = await browser.new_page()

        print("ChatGPT 페이지로 이동합니다. 로그인 안 돼 있으면 직접 로그인해주세요.")
        await ensure_logged_in(page)

        # 로그인 확인을 위해 잠깐 멈춤
        print("로그인이 끝났다면 터미널로 돌아와 엔터를 눌러 진행하세요.")
        try:
            import sys

            sys.stdin.readline()
        except Exception:
            pass

        print("대화 목록을 수집합니다...")
        convs = await fetch_conversations(page, limit=PAGE_SIZE)

        # 최신순 정렬(필요시 제목/생성일 기준으로 바꿀 수 있음)
        # API 응답에 update_time이 들어올 때가 많아 그걸 기준으로 정렬
        def sort_key(c):
            return c.get("update_time") or c.get("create_time") or 0

        convs_sorted = sorted(convs, key=sort_key, reverse=True)

        # 삭제 대상 제한
        if MAX_TO_DELETE is not None:
            convs_sorted = convs_sorted[:MAX_TO_DELETE]

        print(f"가져온 대화 수: {len(convs)}")
        print(f"삭제 대상 대화 수: {len(convs_sorted)} (DRY_RUN={DRY_RUN})")

        # 목록 미리보기
        preview = []
        for i, c in enumerate(convs_sorted, 1):
            title = c.get("title") or "(제목 없음)"
            cid = c.get("id") or c.get("conversation_id")  # 필드명 가변성 대비
            preview.append(f"{i}. {title}  |  id={cid}")
        print("\n".join(preview[:30]))
        if len(preview) > 30:
            print(f"... 그 외 {len(preview) - 30}개")

        if DRY_RUN:
            print("DRY_RUN 모드이므로 실제 삭제는 수행하지 않습니다.")
            await browser.close()
            return

        # 실제 삭제
        deleted, failed = 0, 0
        for i, c in enumerate(convs_sorted, 1):
            cid = c.get("id") or c.get("conversation_id")
            title = c.get("title") or "(제목 없음)"
            if not cid:
                print(f"[{i}] ID를 찾을 수 없어 건너뜀: {title}")
                failed += 1
                continue

            ok = await delete_conversation(page, cid)
            if ok:
                deleted += 1
                print(f"[{i}] 삭제 완료: {title}")
            else:
                failed += 1
                print(f"[{i}] 삭제 실패: {title}")

            if SLEEP_BETWEEN_DELETES and SLEEP_BETWEEN_DELETES > 0:
                await page.wait_for_timeout(int(SLEEP_BETWEEN_DELETES * 1000))

        print(f"완료. 삭제됨: {deleted}개  실패: {failed}개")
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
