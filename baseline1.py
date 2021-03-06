import numpy
import re
import gensim
import keras
import time
import sys
import os
import utils
from keras import backend as KTF
from keras.models import Sequential, Model, load_model
from keras.layers import Input,Concatenate
from keras.layers import Reshape,Dense, Dropout, Embedding, LSTM,Flatten,Conv2D,MaxPooling2D
from keras.optimizers import Adam
TRAINING=int(sys.argv[1])
sent_len=540
esize=300
if(TRAINING):
    KTF.clear_session()
    inp=Input(shape=(sent_len*2,esize,1),dtype='float32')
    conv1=Conv2D(96,(2,esize),activation='relu')(inp)
    conv2=Conv2D(96,(3,esize),activation='relu')(inp)
    conv3=Conv2D(96,(4,esize),activation='relu')(inp)
    pool1=MaxPooling2D((sent_len-1,1))(conv1)
    pool2=MaxPooling2D((sent_len-2,1))(conv2)
    pool3=MaxPooling2D((sent_len-3,1))(conv3)
    flat1=Flatten()(pool1)
    flat2=Flatten()(pool2)
    flat3=Flatten()(pool3)
    conc=Concatenate()([flat1,flat2,flat3])
    conc=Dropout(0.25)(conc)
    dense=Dense(256,activation='relu')(conc)
    dense=Dropout(0.5)(dense)
    out=Dense(2,activation='softmax')(dense)
    model=Model(inputs=inp,outputs=out)
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    toffset=0
    foffset=0
    if(len(sys.argv)>2):
        toffset=int(sys.argv[2])
        foffset=int(sys.argv[3])
    utils.train(model,TRAINING,2048,'baseline1','baseline1-test.txt',True,toffset,foffset)
    #model.save('baseline1.h5')
    '''
    model=Sequential()
    model.add(Conv2D(256,(3,esize),activation='relu',padding='valid',input_shape=(sent_len,esize,1)))
    #print(model.output_shape)
    model.add(MaxPooling2D((sent_len-2,1)))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dense(2,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    '''
else:
    min_loss=1000
    min_pos=-1
    for i in range(1,10):
        model=load_model('baseline1_%d.h5'%i)
        model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        ans=utils.test(model,True)
        print(' '.join(ans))
        if(float(ans[3])<min_loss):
            min_loss=float(ans[3])
            min_pos=i
    os.system('cp baseline1_%d.h5 baseline1.h5'%min_pos)

    
