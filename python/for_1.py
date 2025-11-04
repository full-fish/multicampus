# data = []
# for i in range(5):
#     data.append(int(input("숫자 입력: ")))
# print("입력된 데이터: ", data)


# name = input("이름: ")
# numArr = []
# total = 0
# for i in range(3):
#     num = int(input(f"점수{i+1}: "))
#     total += num
# print(f"총점: {total}")
# print(f"평균: {total/3}")

# numList = []
# for i in range(10):
#     numList.append(int(input("숫자를 입력해주세요: ")))
# envnList = list(filter(lambda x: not (x % 2), numList))
# oddList = list(filter(lambda x: (x % 2), numList))

# evenSum = sum(envnList)
# oddSum = sum(oddList)

# evenAge = evenSum / len(envnList)
# oddAge = oddSum / len(oddList)

# print("짝수 합:", evenSum)
# print("홀수 합:", oddSum)
# print("짝수 평균:", evenAge)
# print("홀수 평균:", oddAge)


even_sum = 0
odd_sum = 0
even_count = 0
odd_count = 0

for i in range(10):
    n = int(input("숫자를 입력해주세요: "))
    if not (n % 2):
        even_sum += n
        even_count += 1
    else:
        odd_sum += n
        odd_count += 1

even_avg = even_sum / even_count
odd_avg = odd_sum / odd_count
print("짝수 합:", even_sum)
print("홀수 합:", odd_sum)
print("짝수 평균:", even_avg)
print("홀수 평균:", odd_avg)
