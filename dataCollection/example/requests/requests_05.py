import requests

url = "https://www.python.org/static/img/python-logo.png"
response = requests.get(url)
if response.status_code == 200:
    with open("python_logo.png", "wb") as f:
        f.write(response.content)
    print("이미지 저장 완료")
else:
    print("이미지 다운 실패")
    print("오류코드: ", response.status_code)
