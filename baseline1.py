import numpy
import re
import gensim
import keras
import time
import sys
import os
import utils
from keras.models import Sequential, Model, load_model
from keras.layers import Input,Concatenate
from keras.layers import Reshape,Dense, Dropout, Embedding, LSTM,Flatten,Conv2D,MaxPooling2D
from keras.optimizers import Adam
TRAINING=int(sys.argv[1])
sent_len=700
esize=300
utils.set_sentlen(sent_len)
tt=time.clock()
if(TRAINING):
    model=Sequential()
    model.add(Conv2D(256,(3,esize),activation='relu',padding='valid',input_shape=(sent_len,esize,1)))
    #print(model.output_shape)
    model.add(MaxPooling2D((sent_len-2,1)))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dense(2,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model=utils.train(model,20,512,'baseline1.txt','baseline1-test.txt',singular=True)
    model.save('baseline1.h5')
else:
    model=load_model('baseline1.h5')
    print(' '.join(utils.test(model,True)))
print(time.clock()-tt)


    
