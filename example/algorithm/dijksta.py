import heapq as hq


def dijkstra(graph, start_node):
    # 1. 거리 정보를 저장할 딕셔너리 초기화
    # 시작 노드를 제외한 모든 노드의 거리를 무한대로 설정
    distances = {node: float("inf") for node in graph}
    distances[start_node] = 0

    # 2. 우선순위 큐(Min Heap) 초기화
    # (거리, 노드) 형태로 저장하여 거리가 가장 짧은 노드부터 처리
    # 힙의 첫 요소는 (노드까지의 거리, 노드)
    heap = [(0, start_node)]

    # 3. 큐에 요소가 남아있지 않을 때까지 반복
    while heap:
        # print("-----------")
        # 가장 짧은 거리부터 처리
        current_distance, current_node = hq.heappop(heap)

        # 이미 처리되었거나, 현재 계산된 거리보다 이미 저장된 거리가 더 짧다면 무시
        if current_distance > distances[current_node]:
            print("skip")  # 이 부분은 출력 코드이므로 주석 처리하거나 제거 가능
            continue

        # 현재 노드와 연결된 인접 노드를 확인
        for next, weight in graph[current_node]:
            # 새로운 경로를 계산: (현재 노드까지의 거리 + 현재에서 인접 노드까지의 가중치)
            distance = current_distance + weight

            # 새로운 경로가 기존의 최단 거리보다 짧다면 갱신
            if distance < distances[next]:
                distances[next] = distance
                # 갱신된 거리와 인접 노드를 우선순위 큐에 추가
                hq.heappush(heap, (distance, next))
                # print("push", distance, neighbor) # 이 부분은 출력 코드이므로 주석 처리하거나 제거 가능

    return distances


graph = {
    "A": [("B", 1), ("C", 4)],
    "B": [("C", 2), ("D", 5)],
    "C": [("D", 1)],  # C에서 D로 가는 경로와 I로 가는 경로
    "D": [("E", 3)],
    "E": [],  # 종착 노드는 비어있는 리스트
}

start_node = "A"
shortest_distances = dijkstra(graph, start_node)

print(f"시작 노드: {start_node}")
print("최단 거리 결과:")
# 결과: {'A': 0, 'B': 1, 'C': 3, 'D': 4, 'E': 7, 'I': 4}
print(shortest_distances)


# def solution(numbers):
#     result = [[-1] for in range(len(numbers))]
