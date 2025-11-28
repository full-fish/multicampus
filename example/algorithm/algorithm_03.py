def binary_search_iterative(data_list, target):
    # 탐색 범위의 시작(low)과 끝점(high)을 설정
    low = 0
    high = len(data_list) - 1

    # low가 high보다 작거나 같을 때까지 반복
    while low <= high:
        # 1. 중간 지점(mid)을 계산
        mid = (low + high) // 2

        # 2. 중간 지점의 값이 목표 값과 일치하는 경우
        if data_list[mid] == target:
            # 인덱스를 반환하고 종료
            return mid

        # 3. 중간 지점의 값이 목표 값보다 작은 경우
        # 목표 값은 중간 값의 오른쪽(큰 쪽)에 존재
        elif data_list[mid] < target:
            # 탐색 범위를 mid의 오른쪽으로 좁힘
            low = mid + 1

        # 4. 중간 지점의 값이 목표 값보다 큰 경우
        # 목표 값은 중간 값의 왼쪽(작은 쪽)에 존재
        else:
            # 탐색 범위를 mid의 왼쪽으로 좁힘
            high = mid - 1

    # 반복문이 종료될 때까지 찾지 못하면 -1을 반환
    return -1


# --- 테스트 코드 ---

# 이진 탐색은 반드시 정렬된 리스트를 사용
my_list = [1, 4, 8, 9, 11, 15, 20]
target_1 = 11
target_2 = 10

# 11을 탐색 (찾음)
index_1 = binary_search_iterative(my_list, target_1)
print(f"리스트: {my_list}")
print(f"목표 값 {target_1}의 인덱스: {index_1}")
# 출력: 목표 값 11의 인덱스: 4

# 10을 탐색 (못 찾음)
index_2 = binary_search_iterative(my_list, target_2)
print(f"목표 값 {target_2}의 인덱스: {index_2}")
# 출력: 목표 값 10의 인덱스: -1

import time

big_list = []
for i in range(100000000):
    big_list.append(i)

target = 99999998

start_time = time.time()
# [측정 대상 작업]이 여기에 누락되어 있습니다.
linear(
    big_list,
)
end_time = time.time()
result = end_time - start_time
print("result: ", result)
