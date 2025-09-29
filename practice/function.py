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


def cal(num1, num2, type):
    if type == 1:
        return num1 + num2
    elif type == 2:
        return num1 - num2
    elif type == 3:
        return num1 * num2
    else:
        return num1 / num2


def main():
    while 1:
        type = float(input("연산을 선택하세요(덧셈: 1, 뺄셈: 2, 곱셈: 3, 나눗셈: 4)"))
        if type not in [1, 2, 3, 4]:
            print("1~4중에 하나를 입력해주세요")
            continue
        num1 = float(input("연산에 참여할 숫자1: "))
        num2 = float(input("연산에 참여할 숫자2: "))
        print(f"결과 값: {cal(num1,num2,type)}")
        break


main()
