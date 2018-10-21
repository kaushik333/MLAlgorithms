from knn import KNN
import numpy as np
import datasets
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# load data
x_train, y_train, x_test, y_test = datasets.gaussian_dataset(n_train=500, n_test=500)
accuracy = list()
for i in range(1, 50, 5):
    model = KNN(k=i+1)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy.append(np.mean(y_pred == y_test))

ax = plt.figure().gca()
plt.plot(range(1, 50, 5), np.asarray(accuracy))
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel('Value of k --->')
plt.ylabel('Accuracy --->')
plt.title('KNN Accuracy v/s iteration curve')
plt.legend()
plt.show()

# model = KNN(k=3)
# model.fit(x_train, y_train)
#
# y_pred = model.predict(x_test)
# print("knn accuracy: " + str(np.mean(y_pred == y_test)))
