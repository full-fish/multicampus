import asyncio
import aiomysql


async def query_user(pool, user_id):
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT * FROM users WHERE id=%s", (user_id,))
            return await cur.fetchone()


async def main():
    pool = await aiomysql.create_pool(
        host="127.0.0.1",
        port=3306,
        user="root",
        password="sktkfka5",
        db="testdb",
        minsize=1,
        maxsize=5,  # 최대 동시 연결 수
    )
    tasks = [query_user(pool, i) for i in range(1, 6)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    print(results)

    pool.close()
    await pool.wait_closed()


asyncio.run(main())
