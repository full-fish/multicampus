# def is_range_check(n):
#     if 0 <= n <= 100:
#         return True
#     else:
#         return False

# def range(n):
#     if 90 <= n:
#         return "A"
#     elif 80 <= n:
#         return "B"
#     else:
#         return "C"

# def main():
#     while 1:
#         score = int(input("0~100까지의 점수를 입력해주세요"))
#         if is_range_check(score):
#             print(f"{range(score)}등급입니다")
#             break
#         else:
#             print("다시 입력해주세요")

# main()

# ----------------------------------------


# def cal(num1, num2, type):
#     if type == 1:
#         return num1 + num2
#     elif type == 2:
#         return num1 - num2
#     elif type == 3:
#         return num1 * num2
#     else:
#         return num1 / num2


# def main():
#     while 1:
#         type = float(input("연산을 선택하세요(덧셈: 1, 뺄셈: 2, 곱셈: 3, 나눗셈: 4)"))
#         if type not in [1, 2, 3, 4]:
#             print("1~4중에 하나를 입력해주세요")
#             continue
#         num1 = float(input("연산에 참여할 숫자1: "))
#         num2 = float(input("연산에 참여할 숫자2: "))
#         print(f"결과 값: {cal(num1,num2,type)}")
#         break

# main()


# def spam(eggs):
#     eggs.append(1)
#     eggs = [2, 3]
#     print("eggs", eggs, "ham", ham)
#     eggs.append(100)
#     print("eggs", eggs, "ham", ham)


# ham = [0]
# spam(ham)
# print(ham)


# def calculate(x, y):
#     total = x + y  # 새로운 값이 할당되어 함수 내부 total은 지역 변수가 됨
#     print("In Function")
#     print("a:", str(a), "b:", str(b), "a + b:", str(a + b), "total:", str(total))
#     return total


# a = 5  # a와 b는 전역 변수
# b = 7
# total = 0  # 전역 변수 total
# print("In Program - 1")
# print("a:", str(a), "b:", str(b), "a + b:", str(a + b))

# sum = calculate(a, b)
# print("After Calculation")
# print("Total:", str(total), " Sum:", str(sum))
# # 지역 변수는 전역 변수에 영향을 주지 않음

# ----------------------------------------


# def asterisk_test_2(*args):
#     x, y, *z = args
#     return x, y, z


# print(asterisk_test_2(3, 4, 5, 6))

# ----------------------------------------


# def avg(*args):
#     print(args)
#     arr = list(args)
#     min_num = min(arr)
#     max_num = max(arr)
#     arr.remove(min_num)
#     arr.remove(max_num)
#     return sum(arr) / len(arr)


# print(avg(1, 2, 3, 4, 5))

# ----------------------------------------

# pt = (37.5665, 126.9780)  # 서울 좌표
# print(pt[0], pt[1])  # 37.5665 126.9780

# d = {pt: "Seoul"}  # 가능 (튜플은 해시 가능)

# print(d)
# print(d[pt])

# a = {}
# print(a)
# a["b"] = 1
# print(a["b"])

# ----------------------------------------


# def f(**args):
#     print(args)
#     print(args.keys())
#     print(list(args.keys()))
#     print(args.values())
#     print(args.items())
#     print(args.get("name"))


# f(name="man", age=25)

# ----------------------------------------


# def test(*args, **kwargs):
#     sum_num = 0
#     avg_num = 0
#     max_num = 0
#     min_num = 0
#     is_show_sum = kwargs.get("show_sum", False)
#     print(is_show_sum)
#     is_show_avg = kwargs.get("show_avg", False)
#     is_show_max = kwargs.get("show_max", False)
#     is_show_min = kwargs.get("show_min", False)
#     if is_show_sum:
#         sum_num = sum(list(args))
#     if is_show_avg:
#         avg_num = sum_num / (len(list(args)))
#     print(sum_num)
#     print(avg_num)


# test(10, 20, 30, show_sum=True, show_avg=True, show_max=True, show_min=True)

# ----------------------------------------

print("abc\tab")
