import requests, json

url = "https://api.github.com/repos/psf/requests"
response = requests.get(url)
data = response.json()  # JSON → dict
# JSON 문자열로 변환
json_str = json.dumps(data, indent=4, ensure_ascii=False)
print(json_str[:200])
# 파일로 저장
with open("repo_info.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
print("JSON 파일 저장 완료")
