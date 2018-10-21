from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

class Sequential(object):
    def __init__(self, layers, loss):
        """
        Sequential model

        Implements a sequence of layers

        Parameters
        ----------
        layers : list of layer objects
        loss : loss object
        """
        self.layers = layers
        self.loss = loss

    def forward(self, x, target=None): #correction
        """
        Forward pass through all layers
        
        if target is not none, then also do loss layer

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number of features
        target : np.array
            The target data of size number of training samples x number of features (one-hot)

        Returns
        -------
        np.array
            The output of the model
        """
        into = np.copy(x)
        for i in range(len(self.layers)):
            out = self.layers[i].forward(into)
            into = np.copy(out)
#            print(out)
        if target is None:
            return into
        else:
            los = self.loss.forward(into,target)
            return los

    def backward(self):
        """
        Compute "backward" computation of fully connected layer

        Returns
        -------
        np.array
            The gradient at the input

        """
        into = self.loss.backward()
#        print(into)
        for i in range(len(self.layers)-1,-1,-1):
            out = self.layers[i].backward(into)
            into = np.copy(out)
        return into

    def update_param(self, lr):
        """
        Update the parameters with learning rate lr

        Parameters
        ----------
        lr : floating point
            The learning rate
        """
        for i in range(len(self.layers)):
            self.layers[i].update_param(lr)
        
    def fit(self, x, y, x_test, y_test, epochs=10, lr=0.1, batch_size=128):
        """
        Fit parameters of all layers using batches

        Parameters
        ----------
        x : numpy matrix
            Training data (number of samples x number of features)
        y : numpy matrix
            Training labels (number of samples x number of features) (one-hot)
        epochs: integer
            Number of epochs to run (1 epoch = 1 pass through entire data)
        lr: float
            Learning rate
        batch_size: integer
            Number of data samples per batch of gradient descent
        """
        num_batches = x.shape[0]/batch_size
        last_batch = x.shape[0]%batch_size
        if last_batch!=0:
            num_batches+=1
        #tot_batches = num_batches + last_batch
        loss_batch = np.zeros(epochs)
        loss_test = np.zeros(epochs)
        for i in range(epochs):
            print("Epoch no:- ",i)
            loss = 0
            for j in range(0,num_batches): 
                print("Batch no:- ",j)
                if(j==num_batches-1):
                    x_batch = x[batch_size*j:,:]
                    y_batch = y[batch_size*j:,:]
                else:
                    x_batch = x[batch_size*j:batch_size*(j+1),:]
                    y_batch = y[batch_size*j:batch_size*(j+1),:]
                    
                loss += self.forward(x_batch,target=y_batch)
                self.backward()
                self.update_param(lr)
            loss_batch[i] = loss/num_batches
            loss_test[i] = self.forward(x_test,target=y_test)
        return loss_batch, loss_test

    def predict(self, x):
        """
        Return class prediction with input x

        Parameters
        ----------
        x : numpy matrix
            Testing data data (number of samples x number of features)

        Returns
        -------
        np.array
            The output of the model (integer class predictions)
        """
        prediction = self.forward(x,target=None)
#        for i in range(prediction.shape[0]):
#            prediction[i,np.argmax(prediction)]=1
#            prediction[i, np.where(prediction[i]!=np.max(prediction[i]))]=0
            
        pred_class = [np.argmax(prediction[i,:]) for i in range(prediction.shape[0])]
        return np.array(pred_class)
        
