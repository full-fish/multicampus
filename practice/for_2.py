import math

# print("2~100까지의 각 수의 약수의 개수는")
# for num in range(2, 101):
#     count = 0
#     for i in range(1, num + 1):
#         if num % i == 0:
#             count += 1
#     print(f"{num}의 약수의 개수는 {count}")


# print("2~100까지의 각 수의 약수의 개수는")
# for num in range(2, 101):
#     count = 0
#     for i in range(1, int(math.sqrt(num)) + 1):
#         if num % i == 0:
#             if i == num // i:
#                 count += 1
#             else:
#                 count += 2
#     print(f"{num}의 약수의 개수는 {count}")

# ----------------------------------------

# print("2~100까지중 소수의 개수")
# result = 0

# for num in range(2, 101):
#     count = 0
#     for i in range(1, int(math.sqrt(num)) + 1):
#         if num % i == 0:
#             if i == num // i:
#                 count += 1
#             else:
#                 count += 2
#     if count == 2:
#         result += 1
# print(f"소수의 개수는 {result}")


# ----------------------------------------

data = []
count = 1
length = 5
for i in range(length):
    temp_arr = []
    row_sum = 0
    if i != length - 1:
        for j in range(1, length):
            temp_arr.append(count)
            row_sum += count
            count += 1
        temp_arr.append(row_sum)
    else:
        for k in range(length):
            col_sum = 0
            for l in range(length - 1):
                col_sum += data[l][k]
            temp_arr.append(col_sum)
    data.append(temp_arr)

print(data)


# ----------------------------------------

score_1 = [49, 80, 20, 100, 80]
score_2 = [43, 60, 85, 30, 90]
score_3 = [49, 82, 48, 50, 100]
all_score = [score_1, score_2, score_3]
result_list = []

for i in range(len(score_1)):
    sum = 0
    for j in range(3):
        sum += all_score[j][i]
    result_list.append(sum / 3)
print(result_list)
