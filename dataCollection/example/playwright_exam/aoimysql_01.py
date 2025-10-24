import asyncio
import aiomysql


async def main():
    conn = await aiomysql.connect(
        host="localhost", port=3306, user="root", password="sktkfka5", db="testdb"
    )

    async with conn.cursor() as cur:
        await cur.execute("SELECT * FROM users")
        result = await cur.fetchall()
        print(result)

    conn.close()


asyncio.run(main())
