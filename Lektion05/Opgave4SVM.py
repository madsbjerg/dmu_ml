import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
iris_dataset = load_iris()

# Split the data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

from sklearn import svm

# Fit the model
clf = svm.SVC(kernel='linear')
model = clf.fit(X_train, Y_train) #using model for plotting

y_pred = clf.predict(X_test)
print("Ypred: " + str(y_pred))

# Evaluate the model
print("Training scores: {:.2f}".format(clf.score(X_train, Y_train)))
print("Test scores: {:.2f}".format(clf.score(X_test,Y_test)))

pos = np.where(Y_train == 1)
neg = np.where(Y_train == 0)

plt.plot(X_train[pos[0], 0], X_train[pos[0], 1], 'ro')
plt.plot(X_train[neg[0], 0], X_train[neg[0], 1], 'bo')
plt.xlim([min(X_train[:, 0]), max(X_train[:, 0])])
plt.xlim([min(X_train[:, 1]), max(X_train[:, 1])])
plt.show()

