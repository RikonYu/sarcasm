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
from keras.layers import Input,Concatenate, Bidirectional
from keras.layers import Reshape,Dense, Dropout, Embedding, LSTM,Flatten,Conv1D,MaxPooling1D
from keras.optimizers import Adam
TRAINING=int(sys.argv[1])
sent_len=540
esize=300
class baseline3_model:
    def __init__(self):
        self.lstm1=Bidirectional(LSTM(256,return_sequences=True))
        self.lstm2=Bidirectional(LSTM(256,return_sequences=True))
        self.conv1=Conv1D(256,3)
        self.conv2_1=Conv1D(128,3)
        self.conv2_2=Conv1D(128,5)
        self.dense=Dense(2,activation='softmax')
    def forward(self,inp):
        #print(KTF.int_shape(inp))
        inp=Reshape([sent_len*2,-1])(inp)
        #print(KTF.int_shape(inp))
        lstm1=self.lstm1(inp)
        lstm2=self.lstm2(lstm1)
        conv1=self.conv1(lstm2)
        conv2_1=self.conv2_1(conv1)
        conv2_2=self.conv2_2(conv1)
        pool1=MaxPooling1D(sent_len*2-4)(conv2_1)
        pool2=MaxPooling1D(sent_len*2-6)(conv2_2)
        kk=Flatten()(Concatenate()([pool1,pool2]))
        out=self.dense(kk)
        return out
        
layers=baseline3_model()
if(TRAINING):
    KTF.clear_session()
    inp=Input(shape=(sent_len*2,esize,1),dtype='float32')
    out=layers.forward(inp)
    toffset=0
    foffset=0
    if(len(sys.argv)>2):
        toffset=int(sys.argv[2])
        foffset=int(sys.argv[3])
    model=Model(inputs=inp,outputs=out)
    model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
    utils.train(model,TRAINING,2048,'baseline3','baseline3-test.txt',True,toffset,foffset)
else:
    ind=1
    best=1000
    bpos=-1
    while(os.path.isfile('baseline3_%d.h5'%ind)==True):
        model=load_model('baseline3_%d.h5'%ind)
        ans=utils.test(model,singular=True)[1]
        print(' '.join(ans))
        if(ans[1]>best):

