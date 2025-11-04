# n = 1
# sum = 0
# while 1:
#     score = input(f"점수{n} 입력하세요")
#     if score == "exit":
#         break
#     if not (0 <= int(score) <= 100):
#         print("잘못 입력했습니다. 0~100의 값을 입력해주세요")

#     sum += int(score)
#     n += 1
# print(f"입력된 점수의 합은 {sum}이고 평균은 {sum/n}입니다{n-1}")


# num = int(input("구구단 몇단을 계산할까? "))
# print(f"{num}\n구구단 {num}단을 계산한다")
# for i in range(1, num + 1, 1):
#     print(f"{num} * {i} = {num*i}")

# decimal = 10
# result = ""
# while decimal > 0:
#     remainder = decimal % 2
#     decimal //= 2
#     result = str(remainder) + result
# print(result)


# import random

# randon_num = random.randint(1, 100)
# print("숫자를 맞춰 보세요 (1~100)")
# user_num = int(input())
# count = 1
# while randon_num != user_num:
#     count += 1
#     if 0 >= user_num or user_num > 100:
#         print("0~100사이 숫자가 아닙니다")
#     elif user_num > randon_num:
#         print("숫자가 큽니다")
#     else:
#         print("숫자가 작습니다")
#     user_num = int(input())
# print(f"정답입니다 숫자는 {randon_num}입니다\n{count}만에 맞췄습니다")


# while 1:
#     num = int(input("구구단 몇단을 입력할까요"))
#     print(f"구구단 {num}단을 입력합니다.")
#     for i in range(1, 10, 1):
#         print(f"{num} * {i} = {num*i}")
#     if num == 0:
#         print("프로그램을 종료합니다")
#         break

# ----------------------------------------
