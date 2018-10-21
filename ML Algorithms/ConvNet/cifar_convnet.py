#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 19:52:43 2018

@author: kaushik
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from layers.dataset import cifar100
from layers import (ConvLayer, FullLayer, FlattenLayer, MaxPoolLayer,   
                    ReluLayer, SoftMaxLayer, CrossEntropyLayer, Sequential)

(x_train,y_train),(x_test, y_test) = cifar100(1337) 
model = Sequential(layers=(ConvLayer(3,16,3),
                           ReluLayer(),
                           MaxPoolLayer(),
                           ConvLayer(16,32,3),
                           ReluLayer(),
                           MaxPoolLayer(),
                           FlattenLayer(),
                           FullLayer(8*8*32,4),
                           SoftMaxLayer()),
                           loss=CrossEntropyLayer())
start_time = time.clock()
lr_vals = [0.1]
losses_train = list()
losses_test = list()
test_acc = np.zeros(len(lr_vals))
for j in range(len(lr_vals)):
    train_loss, test_loss = model.fit(x_train,y_train,x_test,y_test,epochs=8,lr=lr_vals[j],batch_size=128)
    losses_train.append(train_loss)
    losses_test.append(test_loss)
    print("--- RUN TIME %s seconds --- \n" % (time.clock() - start_time))
#    print("Losses are: ",losses)
    plt.plot(range(1,9), losses_train[j], label='Train loss') 
    plt.plot(range(1,9), losses_test[j], label='Test Loss')
    
    predictions_train = model.predict(x_train)
    y_train_class = np.array([np.argmax(y_train[i]) for i in range(y_train.shape[0])])
    print("Train accuracy is: ", np.mean(predictions_train==y_train_class))

    predictions_test = model.predict(x_test)
    y_test_class = np.array([np.argmax(y_test[i]) for i in range(y_test.shape[0])])
    test_acc[j] = np.mean(predictions_test==y_test_class)
    print("Test accuracy is: ", np.mean(predictions_test==y_test_class))
    
plt.xlabel('Epoch Number --->')
plt.ylabel('Loss value --->')
plt.title('Loss v/s epochs curve')
plt.legend()
plt.show()

#plt.figure()
#plt.stem(lr_vals, test_acc)
#plt.xlabel('Learning Rate --->')
#plt.ylabel('Test Accuracy --->')
#plt.title('Accuracy v/s learn-rate curve')
#plt.legend()
#plt.show()


