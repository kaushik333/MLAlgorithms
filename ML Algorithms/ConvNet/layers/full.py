import numpy as np


class FullLayer(object):
    def __init__(self, n_i, n_o):
        """
        Fully connected layer

        Parameters
        ----------
        n_i : integer
            The number of inputs
        n_o : integer
            The number of outputs
        """
        self.x = None
        self.W_grad = None
        self.b_grad = None

        # need to initialize self.W and self.b
        self.W = np.random.normal(0,np.sqrt(2./(n_o + n_i)),(n_o,n_i))
#        self.W = np.ones((n_o,n_i))
        self.b = 0

    def forward(self, x):
        """
        Compute "forward" computation of fully connected layer

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
        self.x : np.array
             The input data (need to store for backwards pass)
        """
        val = np.dot(x,self.W.T) + np.tile(self.b, (x.shape[0], 1))
#        print('full fwd: ',val)
        self.x = x
        return val
        
    def backward(self, y_grad):
        """
        Compute "backward" computation of fully connected layer

        Parameters
        ----------
        y_grad : np.array
            The gradient at the output

        Returns
        -------
        np.array
            The gradient at the input

        Stores
        -------
        self.b_grad : np.array
             The gradient with respect to b (same dimensions as self.b)
        self.W_grad : np.array
             The gradient with respect to W (same dimensions as self.W)
        """
        
        ############
        ##W_grad
        ############
        
        x_val = np.copy(self.x) 
        Wgrad = 0
        for i in range(0,x_val.shape[0]):
            val = np.reshape(x_val[i,:], (-1, x_val.shape[1]))
            yval = np.reshape(y_grad[i,:], (-1, y_grad.shape[1]))
            Wgrad += np.dot(yval.T,val)
#        print(Wgrad.shape)
        self.W_grad = Wgrad
        
        ############
        ##b_grad
        ############        
        self.b_grad = np.array(np.sum(y_grad,axis=0))
        self.b_grad = np.reshape(self.b_grad, (-1,self.b_grad.shape[0]))
        
        ############
        ##x_grad
        ############
        x_val1 = np.copy(y_grad) 
#        print('ygrad is: ',y_grad.shape)
#        xgrad = list()
#        for i in range(0,x_val1.shape[0]):
#             val = np.reshape(x_val1[i,:], (-1, x_val1.shape[1]))
#             print(val.shape)
#             print(self.W.shape)
#             num_val = np.dot(val,self.W)
#             num = np.reshape(num_val, (-1, len(num_val)))
#             print(num.shape)
#             xgrad.append(num)
##        print(xgrad)
#        xg = np.array(xgrad)
#        print(xg[:,:,0].shape)     
#        return xg[:,:,0]
        return (np.dot(x_val1,self.W))
        
    def update_param(self, lr):
        """
        Update the parameters with learning rate lr

        Parameters
        ----------
        lr : floating point
            The learning rate

        Stores
        -------
        self.W : np.array
             The updated value for self.W
        self.b : np.array
             The updated value for self.b
        """
        self.b = self.b - lr*self.b_grad
        self.W = self.W - lr*self.W_grad
