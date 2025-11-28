data_list = [5, 2, 8, 1, 3, 0]


def selection_sort(data_list, descending=False):
    n = len(data_list)
    # 1. 리스트의 길이만큼 반복 (마지막 요소는 자동으로 정렬되므로 n-1번 반복)
    for i in range(n - 1):
        # 현재 정렬되지 않은 부분에서 가장 작은 요소의 인덱스를 저장
        target_index = i  # 시작 값 = 현재 정렬하고자 하는 위치 인덱스
        # 2. 정렬되지 않은 나머지 부분(i+1부터 끝까지)에서 최소값 찾기
        for j in range(i + 1, n):
            if descending:
                if data_list[j] > data_list[target_index]:
                    target_index = j
            else:
                if data_list[j] < data_list[target_index]:
                    target_index = j
                    # 3. 찾은 최소값(target_index)과 현재 위치(i)의 요소를 교환
        data_list[i], data_list[target_index] = data_list[target_index], data_list[i]

    return data_list


print("\n\nselection_sort\n", selection_sort(data_list))


def insertion_sort(data_list, descending=False):
    n = len(data_list)

    # 1. 두 번째 요소 (인덱스 1)부터 시작하여 리스트 끝까지 반복
    for i in range(1, n):
        # 현재 삽입할 요소 (Key)
        key = data_list[i]

        # key가 삽입될 위치를 찾기 위해, 정렬된 앞부분(i-1부터 0까지)을 탐색
        j = i - 1

        # 2. 앞부분의 요소들을 key와 비교하여, key가 들어갈 공간을 만들기
        # 조건:
        # 1) j가 0 이상이어야 하고 (리스트의 시작을 넘어가지 않도록)
        # 2) 현재 정렬된 요소 data_list[j]가 key와 비교
        #    - 오름차순(not descending)일 경우: data_list[j]가 key보다 크면
        #    - 내림차순(descending)일 경우: data_list[j]가 key보다 작으면
        while j >= 0 and (data_list[j] > key if not descending else data_list[j] < key):
            # 현재 요소를 한 칸 뒤로 밀어넣음 (key를 위한 공간 확보)
            data_list[j + 1] = data_list[j]
            j -= 1

        # 3. while 반복이 끝난 후, j+1 위치가 key가 삽입될 최종 위치
        data_list[j + 1] = key

    return data_list


print("\n\ninsertion_sort\n", insertion_sort(data_list))


def bubble_sort(data_list, descending=False):
    n = len(data_list)

    # 1. Outer Loop: 정렬 과정을 n-1번 반복
    # 매 반복마다 가장 큰 요소가 제 위치 (배열의 끝)로 '버블링'
    for i in range(n - 1):
        # 교환이 발생했는지 확인하는 플래그 (최적화)
        swapped = False

        # 2. Inner Loop: 인접한 두 요소를 비교하고 교환
        # 이미 정렬이 완료된 부분은 제외하고 비교 (n-1-i)
        for j in range(n - 1 - i):
            # 인접한 요소 비교
            if descending:
                if data_list[j] < data_list[j + 1]:
                    data_list[j], data_list[j + 1] = data_list[j + 1], data_list[j]
                    swapped = True
            else:
                if data_list[j] > data_list[j + 1]:
                    data_list[j], data_list[j + 1] = data_list[j + 1], data_list[j]
                    swapped = True

            # 3. 최적화: Inner Loop에서 한 번도 교환이 일어나지 않았다면,
            # 리스트는 이미 정렬된 상태이므로 반복을 종료
            if not swapped:
                break
        return data_list


print("\n\nbubble_sort\n", bubble_sort(data_list))


def quick_sort(data_list, descending=False):
    # 1. 종료 조건 (Base Case): 리스트에 요소가 1개 이하이면 이미 정렬된 것
    if len(data_list) <= 1:
        return data_list

    # 2. 피벗(Pivot) 선택: 리스트의 첫 번째 요소를 피벗으로 사용
    pivot = data_list[0]
    # 리스트의 나머지 요소 (피벗 제외)
    rest_of_list = data_list[1:]

    # 3. 분할 (Partitioning) 조건 변경
    if descending:  # 내림차순 정렬
        # - [left]: 피벗보다 크거나 같은 요소들
        # - [right]: 피벗보다 작은 요소들
        left = [x for x in rest_of_list if x >= pivot]
        right = [x for x in rest_of_list if x < pivot]
    else:  # 오름차순 정렬
        # - [left]: 피벗보다 작거나 같은 요소들
        # - [right]: 피벗보다 큰 요소들
        left = [x for x in rest_of_list if x <= pivot]
        right = [x for x in rest_of_list if x > pivot]

    # 4. 정복 및 결합 (Conquer & Combine):
    # 재귀 호출 시에도 descending 매개변수를 전달해야 합니다.
    return quick_sort(left, descending) + [pivot] + quick_sort(right, descending)


print("\n\nquick_sort\n", quick_sort(data_list))
