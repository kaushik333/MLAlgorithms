import numpy as np


class LogisticRegression(object):
    def __init__(self, n_epochs=10, lr=0.1, l2_reg=0):
        """
        Initialize variables
        """
        self.b = None
        self.w = None
        self.n_epochs = n_epochs
        self.lr = lr
        self.l2_reg = l2_reg

    def forward(self, x):
        """
        Compute "forward" computation of logistic regression

        This will return the squashing function:
        f(x) = 1 / (1 + exp(-(w^T x + b)))

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features

        Returns
        -------
        np.array
            A 1 dimensional vector of the logistic function
        """
        val = list()
        for i in range(0, x.shape[0]):
            val.append(1/float((1 + np.exp(-1*(np.matmul(self.w, x[i].T) + self.b)))))
        return np.atleast_1d(val)

        #raise NotImplementedError

    def loss(self, x, y):
        """
        Return the logistic loss
        L(x) = ln(1 + exp(-y * (w^Tx + b)))

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features
        y : np.array
            The vector of corresponding class labels

        Returns
        -------
        float
            The logistic loss value
        """
        val = list()
        for i in range(0, x.shape[0]):
            val.append(np.log(1 + np.exp(-1*y[i]*(np.matmul(self.w, x[i].T) + self.b))))
        array_sum = (1/float(x.shape[0]))*np.sum(val)
        regularization = 0.5*self.l2_reg*np.matmul(self.w, self.w.T)
        return (array_sum + regularization)[0][0]

    def grad_loss_wrt_b(self, x, y):
        """
        Compute the gradient of the loss with respect to b

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features
        y : np.array
            The vector of corresponding class labels

        Returns
        -------
        float
            The gradient
        """

        val = list()
        for i in range(0, x.shape[0]):
            val.append(-1*y[i]/float(1 + np.exp(y[i]*(np.matmul(self.w, x[i].T) + self.b))))
        return np.atleast_1d((1/float(x.shape[0]))*np.sum(val))

    def grad_loss_wrt_w(self, x, y):
        """
        Compute the gradient of the loss with respect to w

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features
        y : np.array
            The vector of corresponding class labels

        Returns
        -------
        np.array
            The gradient (should be the same size as self.w)
        """
        val = list()
        for i in range(0, x.shape[0]):
            val.append(-y[i]*x[i] / float(1 + np.exp(y[i] * (np.matmul(self.w, x[i].T) + self.b))))
        return np.atleast_1d(((1 / float(x.shape[0])) * np.sum(val, axis=0)) + self.l2_reg*self.w)

    def fit(self, x, y):
        """
        Fit the model to the data

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features
        y : np.array
            The vector of corresponding class labels
        """

        self.w = np.random.rand(1, x.shape[1])
        self.b = 0
        b1 = list()
        w1 = list()
        alpha = self.lr
        for i in range(0, self.n_epochs):
            b1.append(self.b - alpha*self.grad_loss_wrt_b(x, y))
            w1.append(self.w - (alpha * self.grad_loss_wrt_w(x, y)))
            self.b = self.b - (alpha*self.grad_loss_wrt_b(x, y))
            self.w = self.w - (alpha*self.grad_loss_wrt_w(x, y))
        return w1, b1


    def predict(self, x):
        """
        Predict the labels for test data x

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of features

        Returns
        -------
        np.array
            Vector of predicted class labels for every training sample
        """

        prob = self.forward(x)
        prob[prob > 0.5] = 1
        prob[prob <= 0.5] = -1

        return prob