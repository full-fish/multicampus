# import sys  # sys 모듈을 호출

# print(sys.getsizeof("a"), sys.getsizeof("aba"), sys.getsizeof("a"))

# arr = [2, 3, 4]
# # for i in [2, 3, 4]:
# del arr[1:2]
# print(arr)


# f = open("practice/yesterday.txt", "r")
# yesterday_lyric = f.readlines()
# f.close()

# with open("practice/yesterday.txt", "r") as f:
#     yesterday_lyric = f.readlines()
# # print(yesterday_lyric)
# contents = ""
# for line in yesterday_lyric:
#     contents = contents + line.strip() + "\n"
# count = contents.upper().count("YESTERDAY")
# print(count)

# a = "%1d" % 12
# print(a, len(a))
# b = "%3.3f" % 5.94343
# print(b, len(b))

# word = input("Input a word: ")
# world_list = list(word)
# print(world_list)
# result = []
# for _ in range(len(world_list)):
#     result.append(world_list.pop())
# print(result)
# print(word[::-1])
