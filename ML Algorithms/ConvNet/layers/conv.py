import numpy as np
import scipy.signal


class ConvLayer(object):
    def __init__(self, n_i, n_o, h):
        """
        Convolutional layer

        Parameters
        ----------
        n_i : integer
            The number of input channels
        n_o : integer
            The number of output channels
        h : integer
            The size of the filter
        """
        # glorot initialization
        self.W = np.random.normal(0,np.sqrt(2./(n_o + n_i)),(n_o,n_i,h,h))
        self.b = np.zeros(n_o)

        self.n_i = n_i
        self.n_o = n_o
        self.h = h

        self.W_grad = None
        self.b_grad = None

    def forward(self, x):
        """
        Compute "forward" computation of convolutional layer

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of input channels x number of rows x number of columns

        Returns
        -------
        np.array
            The output of the convolutions

        Stores
        -------
        self.x : np.array
             The input data (need to store for backwards pass)
        """
        x_copy = x
        
        ##############
        #padding input
        ##############
        x_new = list()
        for i in range(x.shape[0]):#across all batches
            x_n = list()
            for j in range(x.shape[1]):#across all channels
                x_n.append(np.pad(x_copy[i,j,:,:], ((self.h/2, self.h/2), (self.h/2, self.h/2)), 'constant', constant_values=(0, 0)))
            x_new.append(np.array(x_n))  
        x_new = np.array(x_new)
        
        
        out_batch = list()
        for i in range(x.shape[0]): ##batch size
            res = list()
            for j in range(self.n_o): ##number of filter banks
                result=0
                for k in range(x.shape[1]): #channels
                    result += scipy.signal.correlate(x_new[i,k,:,:], self.W[j,k,:,:], mode='valid')
                res.append(result+ self.b[j])
            res_arr = np.array(res) 
            out_batch.append(res_arr)
        out_batch = np.array(out_batch)
        self.x = x
        
        return out_batch
        

    def backward(self, y_grad):
        """
        Compute "backward" computation of convolutional layer

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
        self.w_grad : np.array
             The gradient with respect to W (same dimensions as self.W
        """
        
        x_copy = self.x
        
        ##############
        #padding input
        ##############
        x_new = list()
        for i in range(self.x.shape[0]):#across all batches
            x_n = list()
            for j in range(self.x.shape[1]):#across all channels
                x_n.append(np.pad(x_copy[i,j,:,:], ((self.h/2, self.h/2), (self.h/2, self.h/2)), 'constant', constant_values=(0, 0)))
            x_new.append(np.array(x_n))  
        x_new = np.array(x_new)
        
        
        ######################
        ### b_grad
        ######################
        
        bgrad = np.zeros(self.b.shape)
        for j in range(y_grad.shape[1]): #across all channels (don't sum)
            sum1=0
            for i in range(y_grad.shape[0]): ##across all batches
                sum1 += np.sum(y_grad[i,j,:,:])
            bgrad[j] = sum1
        self.b_grad = bgrad
        #######################
        ### w_grad
        #######################  
#        print self.x.shape
#        print y_grad.shape
#        x_copy = self.x
        
        w_batch = list()
        for i in range(y_grad.shape[1]): ##across all output channels
            res_batch = 0
            for j in range(y_grad.shape[0]): ##across all batches
                result = list()
                for k in range(x_new.shape[1]): #across all input channels
                    result.append(scipy.signal.correlate(x_new[j,k,:,:], y_grad[j,i,:,:], mode='valid'))
                res_batch += np.array(result)    
            w_batch.append(res_batch)
        wgrad = np.array(w_batch)
        self.W_grad = wgrad
        #######################
        ### x_grad
        #######################
        filt_final = np.zeros(self.x.shape)
        for i in range(y_grad.shape[0]): ##across all batches
            filt = list()
            for k in range(self.W.shape[1]): ##across all filter channels
                result = 0
                for j in range(y_grad.shape[1]): ##across all output channels
                    result+=scipy.signal.convolve(y_grad[i,j,:,:], self.W[j,k,:,:], mode='same')
                filt.append(result)
            filt_final[i,:,:,:] = np.array(filt)     
        
        return np.array(filt_final)

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
