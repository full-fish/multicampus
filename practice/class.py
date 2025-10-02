# class Person:
#     def __init__(self, name, age):
#         self.name = name
#         self.age = age

#     def __del__(self):  # 객체 소멸 시 자동 호출
#         print(f"{self.name} is deleted")

#     def __str__(self):  # print()나 str()에서 사용
#         return f"Person({self.name})"

#     def change_name(self, name):
#         self.name = name

#     def change_age(self, age):
#         self.age = age

#     def show_data(self):
#         print(f"이름: {self.name}")
#         print(f"나이: {self.age}")


# p1 = Person("manseon", 32)
# p1.gender = "F"
# print(p1.name)
# print(p1)
# print(p1.gender)
# p1.show_data()
# p1.change_name("kim")
# p1.show_data()
# print(p1.keys())

# ----------------------------------------


class Person(object):  # 부모 클래스 Person 선언
    def __init__(self, name, age, gender):
        self.name = name
        self.age = age
        self.gender = gender

    def about_me(self):  # 메서드 선언
        print("저의 이름은", self.name, "이고요, 제 나이는", str(self.age), "살입니다.")


class Employee(Person):  # 부모 클래스 Person으로부터 상속
    def __init__(self, name, age, gender, salary, hire_date):
        super().__init__(name, age, gender)  # 부모 객체 사용
        self.salary = salary
        self.hire_date = hire_date  # 속성값 추가

    def do_work(self):  # 새로운 메서드 추가
        print("열심히 일을 한다.")

    def about_me(self):  # 부모 클래스 함수 재정의
        super().about_me()  # 부모 클래스 함수 사용
        print(
            "제 급여는", self.salary, "원이고, 제 입사일은", self.hire_date, "입니다."
        )


p1 = Person(name="manseon", age="32", gender="M")
p1.about_me()
print("----")
p2 = Employee(name="manseon", age="32", gender="M", salary=100, hire_date="2025")
p2.about_me()
