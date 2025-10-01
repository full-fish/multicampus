# gems = ["DIA", "RUBY", "RUBY", "DIA", "DIA", "EMERALD", "SAPPHIRE", "DIA"]
# print((set(gems)))

# dic = dict(aa=100, bb=200)
# print(dic)
# dic.update(cc=300, dd=400)
# print(dic)
# dic.update({"ee": 500, "ff": 600})
# print(dic)

from collections import OrderedDict
from collections import defaultdict
from collections import Counter


def sort_by_key(t):
    return t[0]


from collections import OrderedDict  # OrderedDict 모듈 선언

# d = dict()
# d["x"] = 100
# d["y"] = 200
# d["z"] = 300
# d["l"] = 500
# # changed = OrderedDict(sorted(d.items(), key=sort_by_key)).items()
# changed = OrderedDict(sorted(d.items(), key=lambda t: t[0])).items()
# print(list(changed))

# s = [("yellow", 1), ("blue", 2), ("yellow", 3), ("blue", 4), ("red", 1)]
# d = defaultdict(list)
# print("d", d)
# for k, v in s:
#     print("k", k, "v", v)
#     d[k].append(v)
#     print(d)
# print(d.items())

# arr = [{"a": [1], "b": [2]}]
# arr.append({"a": [2]})
# print(arr)


from collections import Counter

# text = list("gallahad")
# c = Counter(text)
# print(c)
# print(list(c))
# print(list(c.elements()))

c = Counter(a=4, b=2, c=0, d=-2)
d = Counter(a=1, b=2, c=3, d=4)
# c.subtract(d)
print(c)
print(c - d)
