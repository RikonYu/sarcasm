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
def forward(inp):
    conv1_1=Conv2D(50,(4,1),activation='relu',padding='valid')(inp)
    conv1_2=Conv2D(50,(5,1),activation='relu',padding='valid')(inp)
    conv1=Concatenate(axis=1)([conv1_1,conv1_2])
    pool1=MaxPooling2D((2,1))(conv1)
    conv2=Conv2D(100,(3,1),activation='relu',padding='valid')(pool1)
    pool2=MaxPooling2D((2,esize))(conv2)
    flat1=Flatten()(pool2)
    dense1=Dense(100,activation='relu')(flat1)
    dense1=Dropout(0.25)(dense1)
    return dense1
if(TRAINING):
    KTF.clear_session()
    main_inp=Input(shape=(sent_len*2,esize,1),dtype='float32')
    sent_inp=Input(shape=(sent_len*2,esize,1),dtype='float32')
    #main_out=forward(main_inp)
    sent_out=forward(sent_inp)
    main_out=forward(main_inp)
    sent_res=Dense(2,activation='softmax')(sent_out)
    out=Concatenate()([main_out,sent_out])
    out=Dense(2,activation='softmax')(out)
    toffset=0
    foffset=0
    #raise Exception
    if(len(sys.argv)>2):
        toffset=int(sys.argv[2])
        foffset=int(sys.argv[3])
    if(os.path.isfile('baseline3-pr.h5')==False):
        #pretrain
        model=Model(inputs=sent_inp,outputs=sent_res)
        model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        utils.pretrain(model,TRAINING,512,'baseline3',toffset,double=True)
    else:
        model=Model(inputs=[main_inp,sent_inp],outputs=out)
        model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        model.load_weight('baseline3-pr.h5',by_name=True)
        utils.train(model,TRAINING,512,'baseline3','baseline3-test.txt',True,toffset,foffset)
else:
    if(os.path.isfile('baseline3-pr.h5')==False):
        ind=1
        best=1000
        bpos=-1
        while(os.path.isfile('baseline3_pr_%d.h5'%ind)==True):
            model=load_model('baseline3_pr_%d.h5'%ind)
            ans=utils.test(model,singular=True)[1]
            if(ans>best):
                best=ans
                bpos=ind
            ind+=1
        os.system('cp baseline3_pr_%d.h5 baseline3-pr.h5'%min_pos)
    else:
        ind=1
        best=1000
        bpos=-1
        while(os.path.isfile('baseline3_%d.h5'%ind)==True):
            model=load_model('baseline3_%d.h5'%ind)
            ans=utils.test(model,singular=True)[1]
            if(ans>best):
                best=ans
                bpos=ind
            ind+=1
        os.system('cp baseline3_%d.h5 baseline3.h5'%min_pos)
