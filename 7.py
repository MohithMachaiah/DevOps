from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
iris = load_iris()
X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size = 0.2,random_state = 42)
gamma = 0.5
C_values = [0.01,1,10]
best_accuracy = 0
best_C = 0
total_support_vectors = 0
for C in C_values:
    model = SVC(kernel = 'rbf',gamma = gamma,C=C)
    model.fit(X_train,y_train)
    accuracy = accuracy_score(y_test,model.predict(X_test))
    support_vectors = model.n_support_.sum()
    if accuracy>best_accuracy:
        best_accuracy = best_accuracy
        best_C = C
        total_support_vectors = support_vectors
        print(f"Best accuracy:{best_accuracy*100:2f}%")
print(f"Best Cvalue: {best_C}")    
print(f"Total number of support vectors: {total_support_vectors}")