from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

iris = load_iris(as_frame=True)
X = iris.data
Y = iris.target

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(x_train, y_train)

pred = dt.predict(x_test)
print(accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

plt.figure(figsize=(12, 6))
plot_tree(
    dt,
    feature_names=X.columns,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
)
plt.tight_layout()
plt.show()

print(dt.feature_importances_)

for name, imp in zip(X.columns, dt.feature_importances_):
    print(f"{name:20s} {imp:.3f}")
