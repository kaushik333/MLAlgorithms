import numpy as np
from scipy import stats


class KNN(object):
    def __init__(self, k=3):
        self.x_train = None
        self.y_train = None
        self.k = k

    def fit(self, x, y):
        """
        Fit the model to the data

        For K-Nearest neighbors, the model is the data, so we just
        need to store the data

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features
        y : np.array
            The vector of corresponding class labels
        """
        self.x_train = x
        self.y_train = y

    def predict(self, x):
        """
        Predict x from the k-nearest neighbors

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features

        Returns
        -------
        np.array
            A vector of size N of the predicted class for each sample in x
        """
        belong_class = list()
        for i in range(0, x.shape[0]):
            dist = list()
            ind = [None]*self.x_train.shape[0]
            class_labels = [None]*self.x_train.shape[0]
            ind_new = [None] * self.x_train.shape[0]
            for j in range(0, self.x_train.shape[0]):
                dist.append(np.linalg.norm(x[i] - self.x_train[j]))
            ind = np.argsort(np.asarray(dist))
            ind_new = ind[0:self.k]
            class_labels = [self.y_train[num] for num in ind_new]
            belong_class.append(stats.mode(class_labels)[0].tolist()[0])

        return belong_class