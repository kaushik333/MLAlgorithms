from logistic_regression import LogisticRegression
import numpy as np
import datasets
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# load data
x_train, y_train, x_test, y_test = datasets.gaussian_dataset(n_train=500, n_test=500)
loss_val = list()
alphas = [0.01, 0.1, 50]
model = LogisticRegression(n_epochs=100)   
strg = "loss_val_"
my_loss = list()
for j in range(0, len(alphas)):
    model.lr = alphas[j]
    w1, b1 = model.fit(x_train, y_train)
    vars()[strg + str(j)] = list()
    for i in range(0, len(w1)):
        model.w = w1[i]
        model.b = b1[i]
        vars()[strg + str(j)].append(model.loss(x_train, y_train))
    my_loss.append(vars()[strg + str(j)])

ax = plt.figure().gca()
for i in range(0, len(alphas)):
    string = 'alpha='+str(alphas[i])
    plt.plot(range(1, model.n_epochs+1), my_loss[i], label=string)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel('Iteration number --->')
plt.ylabel('Loss value --->')
plt.title('Logistic Regression loss v/s iteration curve')
plt.legend()
plt.show()


model = LogisticRegression(n_epochs=100)
w2, b2 = model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("logistic regression: " + str(np.mean(y_pred == y_test)))
