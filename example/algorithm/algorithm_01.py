class Node:
    # 노드 객체를 생성하고 데이터(data)와 다음 노드 포인터(next)를 초기화합니다.
    def __init__(self, data):
        self.data = data  # 노드가 저장할 실제 데이터
        self.next = None  # 다음 노드를 가리키는 포인터 (초기에는 None)


class LinkedList:
    def __init__(self):
        self.head = None

    # 비어있나?
    def isEmpty(self):
        return self.head is None

    # 전체 길이
    def length(self):
        count = 0
        current = self.head
        while current:
            count += 1
            current = current.next
        return count

    # 삽입
    def insert(self, index, data):
        new_node = Node(data)

        if index == 0:  # head에 삽입
            new_node.next = self.head
            self.head = new_node
            return

        prev = self.head
        for _ in range(index - 1):
            if prev is None:
                raise IndexError("Index out of range")
            prev = prev.next
        new_node.next = prev.next
        prev.next = new_node

    def printList(self):
        current = self.head
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")

    # 삭제
    def delete(self, index):
        # List가 비어있는지 확인
        if self.head is None:
            raise IndexError("List is empty")

        # 헤드를 삭제할 경우
        if index == 0:
            self.head = self.head.next
            return

        # # 해당 index가 정상인지 확인 (삭제할 노드의 앞 노드(prev) 찾기)
        prev = self.head
        for _ in range(index - 1):
            if prev.next is None:
                raise IndexError("Index out of range")
            prev = prev.next

        # prev.next가 None인 경우 (삭제하려는 노드가 없거나, 인덱스가 리스트 길이를 초과함)
        if prev.next is None:
            raise IndexError("Index out of range")

        # 삭제 후, 앞 - 뒤 연결
        prev.next = prev.next.next

    def get(self, index):
        current = self.head
        for _ in range(index):
            if current is None:
                raise IndexError("Index out of range")
            current = current.next

        if current is None:
            raise IndexError("Index out of range")

        return current.data

    def update(self, index, data):
        current = self.head
        for _ in range(index):
            if current is None:
                raise IndexError("Index out of range")
            current = current.next

        if current is None:
            raise IndexError("Index out of range")

        current.data = data


print("--- 연결 리스트 테스트 시작 ---")

# 1. LinkedList 객체 생성
my_list = LinkedList()
print(f"리스트 생성 후 비어있나? {my_list.isEmpty()}")

# 2. insert 메서드 테스트
print("\n--- 삽입 테스트 ---")
# 맨 앞에 삽입 (인덱스 0)
my_list.insert(0, 10)
my_list.insert(0, 5)
# 중간에 삽입 (인덱스 1)
my_list.insert(1, 7)
# 맨 끝에 삽입 (현재 길이 3)
my_list.insert(3, 15)
# 리스트 출력
my_list.printList()  # 예상 출력: 5 -> 7 -> 10 -> 15 -> None

# 4. length 메서드 테스트
print(f"현재 리스트의 길이: {my_list.length()}")  # 예상 출력: 4
