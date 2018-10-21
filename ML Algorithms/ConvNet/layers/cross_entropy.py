import numpy as np


class CrossEntropyLayer(object):
    def __init__(self):
        """
        Constructor
        """
        self.x = None
        self.t = None

    def forward(self, x, t):
        """
        Implements forward pass of cross entropy

        l(x,t) = 1/N * sum(log(x) * t)

        where
        x = input (number of samples x feature dimension)
        t = target with one hot encoding (number of samples x feature dimension)
        N = number of samples (constant)

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x feature dimension
        t : np.array
            The target data (one-hot) of size number of training samples x feature dimension

        Returns
        -------
        np.array
            The output of the loss

        Stores
        -------
        self.x : np.array
             The input data (need to store for backwards pass)
        self.t : np.array
             The target data (need to store for backwards pass)
        """
        batch_size = x.shape[0]
        #x_eval = np.copy(x)
        #x_eval[np.where(x==0)]=1
        loss = (-1./batch_size)*np.sum(np.multiply(t,np.log(x)))

        self.x = x
        self.t = t
        return loss
        
        #raise NotImplementedError

    def backward(self, y_grad=None):
        """
        Compute "backward" computation of softmax loss layer

        Returns
        -------
        np.array
            The gradient at the input

        """
        batch_size = self.x.shape[0]
        num_feat = self.x.shape[1]
        y_g = list()
#        t_v = np.copy(self.t)
#        t_v[np.where(self.x==0)]=0
#        x_v = np.copy(self.x)
#        x_v[np.where(self.x==0)]=1
        for i in range(0,batch_size):
            val = [(-1./batch_size)*(self.t[i,j]/self.x[i,j]) for j in range(num_feat)]
#            print(val)
            y_g.append(val)
            
        return np.array(y_g)
            
        #raise NotImplementedError
