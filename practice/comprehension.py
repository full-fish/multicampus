# print([i + 10 for i in range(10)])

# words = "The quick brown fox jumps over the lazy dog".split()
# stuff = [[w.upper(), w.lower(), len(w)] for w in words]
# print(stuff)

# import time

# iteration_max = 10000
# vector = list(range(iteration_max))
# scalar = 2
# start = time.perf_counter()
# for _ in range(iteration_max):
#     result = []
#     for value in vector:
#         result.append(scalar * value)
# end = time.perf_counter()
# print("for문 사용 경과 시간: ", end - start)

# start = time.perf_counter()
# for _ in range(iteration_max):
#     [scalar * value for value in range(iteration_max)]
# end = time.perf_counter()
# print("Comprehenstion문 사용 경과 시간: ", end - start)


# import time

# n = 10_000
# vector = list(range(n))
# scalar = 2

# # for + append
# t0 = time.perf_counter()
# for _ in range(n):
#     result = []
#     for v in vector:
#         result.append(scalar * v)
# t1 = time.perf_counter()

# # comprehension (같은 vector 사용)
# for _ in range(n):
#     result = [scalar * v for v in vector]
# t2 = time.perf_counter()

# print("for+append:", t1 - t0)
# print("comprehension:", t2 - t1)

# for i in (1, 2, 3):
#     print(i)


# print(
#     {
#         {i: j}
#         for i, j in enumerate(
#             "TEAMLAB is an academic institute located in South Korea.".split()
#         )
#     }
# )

# names = ["aaa", "bbb", "ccc"]
# score = [100, 80, 60]
# print(dict((["aaa", 100], ["bbb", 200])))
# d1 = dict(zip(names, score))
# print(d1)
# dic_score = dict(aa=80, bb=70, cc=40, dd=30)
# result = {i: ("pass" if v >= 70 else "false") for i, v in dic_score.items()}
# print(result)
ex = [1, 2, 3, 4]
a = [x**2 if x % 2 == 0 else x for x in ex]
b = list(map(lambda x: x**2, filter(lambda x: x % 2 == 0, ex)))
print(a)
print(b)
