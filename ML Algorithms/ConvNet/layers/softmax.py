import numpy as np


class SoftMaxLayer(object):
    def __init__(self):
        """
        Constructor
        """
        self.y = None

    def forward(self, x):
        """
        Implement forward pass of softmax

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number of features

        Returns
        -------
        np.array
            The output of the layer

        Stores
        -------
        self.y : np.array
             The output of the layer (needed for backpropagation)
        """
        soft_x = list()
        k2 = x - np.max(x)
        for i in range(x.shape[0]):
            k = np.copy(k2[i])
            k1 = [np.exp(k[j])/np.sum(np.exp(k)) for j in range(0,len(k))]
            soft_x.append(k1)
        self.y = np.array(soft_x)
        return self.y

    def backward(self, y_grad):
        """
        Compute "backward" computation of softmax

        Parameters
        ----------
        y_grad : np.array
            The gradient at the output

        Returns
        -------
        np.array
            The gradient at the input

        """
        z = np.copy(self.y)
        soft_back = list()
        for i in range(z.shape[0]):
            l = z[i]
            jacobian = np.diag(l) - np.outer(l,l)
            soft_back.append(np.dot(y_grad[i],jacobian))
            
        return np.array(soft_back)
    
    def update_param(self, lr):
        pass  # no learning for softmax layer
