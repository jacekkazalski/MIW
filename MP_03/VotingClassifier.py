import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.svm import SVC

X, y = make_moons(n_samples=10000, noise=0.4, random_state=10)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

svm_clf = SVC(kernel='linear', random_state=0)
log_reg_clf = LogisticRegression(random_state=0)
rf_clf = RandomForestClassifier(random_state=0)

voting_clf = VotingClassifier(estimators=[('lr', log_reg_clf), ('svm', svm_clf), ('rf', rf_clf)])

voting_clf.fit(X_train, y_train)

test_accuracy = voting_clf.score(X_test, y_test)
train_accuracy = voting_clf.score(X_train, y_train)
print(test_accuracy)
print(train_accuracy)

plt.figure(figsize=(10, 10))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='.', label='Train')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='x', label='Test')
xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100),
                     np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100))
Z = voting_clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title(f"Voting Classifier\nTrain acc: {train_accuracy:.4f} Test acc: {test_accuracy:.4f}")
plt.legend()
plt.tight_layout()
plt.show()
