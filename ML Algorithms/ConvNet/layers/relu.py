import numpy as np
class ReluLayer(object):
    def __init__(self):
        """
        Rectified Linear Unit
        """
        self.y = None

    def forward(self, x):
        """
        Implement forward pass of Relu

        y = x if x > 0
        y = 0 otherwise

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
             The output data (need to store for backwards pass)
        """

        x_var = np.copy(x)
        x_var[np.where(x_var<=0)]=0
        self.y = x
        return x_var

    def backward(self, y_grad):
        """
        Implement backward pass of Relu

        Parameters
        ----------
        y_grad : np.array
            The gradient at the output

        Returns
        -------
        np.array
            The gradient at the input
        """
        y_grad[np.where(self.y<=0)]=0                        
        
        return y_grad

    def update_param(self, lr):
        pass  # no parameters to update
