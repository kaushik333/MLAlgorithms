import numpy as np


class SVM(object):
    def __init__(self, n_epochs=10, lr=0.1, l2_reg=1):
        """
        """
        self.b = None
        self.w = None
        self.n_epochs = n_epochs
        self.lr = lr
        self.l2_reg = l2_reg

    def forward(self, x):
        """
        Compute "forward" computation of SVM f(x) = w^T*x + b

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
            val.append((np.matmul(self.w, x[i].T) + self.b)[0])
        return val   
        

    def loss(self, x, y):
        """
        Return the SVM hinge loss
        L(x) = max(0, 1-y(f(x))) + 0.5 w^T w

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
            midval1 = (np.matmul(self.w, x[i].T) + self.b)[0]
            midval = (1 - y[i]*midval1)
            val.append(np.maximum(0, midval))
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
            ###############################
            ### CHECK IF 1 - y(wTx + b) >0
            ###############################
        
            if (1 - y[i]*(np.matmul(self.w, x[i].T) + self.b) > 0):
                #val.append(np.maximum(0, -y[i]))
                val.append(-y[i])
            else:
                val.append(0)
        return (1/float(x.shape[0])*np.sum(val))

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
            
            ###############################
            ### CHECK IF 1 - y(wTx + b) >0
            ###############################
        
            if (1 - y[i]*(np.matmul(self.w, x[i].T) + self.b) > 0):
                #midval = [np.zeros(x.shape[1]), -y[i]*x[i]]
                #num = (1/float(x.shape[0]))*np.amax(midval, axis=0)
                #val.append(np.atleast_1d(num))
                val.append((1/float(x.shape[0]))*-y[i]*x[i])
            else:
                val.append(np.zeros(x.shape[1]))          
            
#            midval = [np.zeros(x.shape[1]), -y[i]*x[i]]
#            num = (1/float(x.shape[0]))*np.amax(midval, axis=0)
#            val.append(np.atleast_1d(num))
        return np.atleast_1d(np.sum(val, axis=0) + self.l2_reg*self.w)

    def fit(self, x, y, plot=False):
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
        for i in range(0, self.n_epochs):
            b1.append(self.b - (self.lr*self.grad_loss_wrt_b(x, y)))
            w1.append(self.w - (self.lr * self.grad_loss_wrt_w(x, y)))
            b_val = self.b - (self.lr*self.grad_loss_wrt_b(x, y))
            w_val = self.w - (self.lr*self.grad_loss_wrt_w(x, y))
            self.b = b_val
            self.w = w_val
            #print self.w
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
        num = np.atleast_1d(self.forward(x))
        num[num > 0] = 1
        num[num <= 0] = -1


        return num
