from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(C=1e4, max_iter=200)

model.fit(X_train, y_train)

accuracy = accuracy_score(y_test,model.predict(X_test))
print(f"Accuracy:{accuracy*100:.2f}%")

