import numpy as np


class MaxPoolLayer(object):
    def __init__(self, size=2):
        """
        MaxPool layer
        Ok to assume non-overlapping regions
        """
        self.locs = None  # to store max locations
        self.size = size  # size of the pooling

    def forward(self, x):
        """
        Compute "forward" computation of max pooling layer

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number
            of input channels x number of rows x number of columns

        Returns
        -------
        np.array
            The output of the maxpooling

        Stores
        -------
        self.locs : np.array
             The locations of the maxes (needed for back propagation)
        """
        stride = self.size
        
        output_size_x = (x.shape[2] - self.size)/stride + 1
        output_size_y = (x.shape[3] - self.size)/stride + 1
        x_reduced = np.zeros([x.shape[0], x.shape[1], output_size_x, output_size_y])
        max_loc = np.zeros(x.shape)
        for i in range(x.shape[0]): #across all batches
            for j in range(x.shape[1]): #across all channels
                x_channel = x[i,j,:,:]
                
                ##############
                #MAXPOOL
                ##############
                
                x_win = np.repeat(range(stride),stride,axis = 0)
                y_win = np.tile(np.array(range(stride)), stride)
                
                window = [x_win, y_win]
                
                ind_x = ind_y = 0
                while(np.max(window[0])<x.shape[2]):
                    while(np.max(window[1])<x.shape[3]):
                        max_val = np.max(x_channel[window])
                        x_reduced[i,j,ind_x,ind_y] = max_val
                        max_list = [p for p, q in enumerate(x_channel[window]) if q == max_val]
                        for index in max_list:
                            max_loc[i,j,window[0][index],window[1][index]] = 1
                        ind_y+=1
                        window[1]+=stride
                    ind_x+=1
                    ind_y=0
                    window[1]=np.tile(np.array(range(stride)), stride)
                    window[0]+=stride
        self.locs = max_loc
        return x_reduced
                
                
    def backward(self, y_grad):
        """
        Compute "backward" computation of maxpool layer

        Parameters
        ----------
        y_grad : np.array
            The gradient at the output

        Returns
        -------
        np.array
            The gradient at the input
        """
        stride = self.size
        
        for i in range(self.locs.shape[0]): #across all batches
            for j in range(self.locs.shape[1]): #across all channels
                
                current_frame = np.copy(self.locs[i,j,:,:])
                x_win = np.repeat(range(stride),stride,axis = 0)
                y_win = np.tile(np.array(range(stride)), stride)
                
                window = [x_win, y_win]
                ind_x = ind_y = 0
                
                while(np.max(window[0])<self.locs.shape[2]):
                    while(np.max(window[1])<self.locs.shape[3]):
                        indices = np.where(current_frame[window]==1)[0]
                        for k in indices:
                            self.locs[i, j, window[0][k],window[1][k]] = y_grad[i,j,ind_x,ind_y]
                        ind_y+=1
                        window[1]+=stride
                    ind_x+=1
                    ind_y=0
                    window[1]=np.tile(np.array(range(stride)), stride)
                    window[0]+=stride   
            
        return self.locs
        
        

    def update_param(self, lr):
        pass
