import asyncio, time


# async def worker(name):
#     print("시작", name)
#     await asyncio.sleep(1)
#     print("완료", name)


# async def main():
#     tasks = [worker("A"), worker("B"), worker("C")]
#     print(tasks)
#     await asyncio.gather(*tasks)

# asyncio.run(main())


# ---------------------------------------- 성능 재기 비동기


# async def say_hello(name, delay):
#     await asyncio.sleep(delay)
#     return f"hello {name}"


# async def main():
#     start = time.perf_counter()
#     tasks = [say_hello("man", 2), say_hello("minji", 1), say_hello("yuri", 4)]
#     result = await asyncio.gather(*tasks)
#     end = time.perf_counter()
#     print(result)
#     print("소요 시간", end - start)


# asyncio.run(main())


# ---------------------------------------- 성능 재기 동기


def say_hello(name, delay):
    time.sleep(delay)
    return f"hello {name}"


def main():

    result = []
    start = time.perf_counter()

    result.append(say_hello("man", 2))
    result.append(say_hello("minji", 1))
    result.append(say_hello("yuri", 4))

    end = time.perf_counter()
    print(result)
    print("소요 시간", end - start)


main()
