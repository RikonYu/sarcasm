import numpy
import re
import gensim
import keras
import time
import sys
import nltk
import os
import utils
from keras import backend as KTF
from keras.models import Sequential, Model, load_model
from keras.layers import Input,Concatenate, Layer
from keras.layers import Reshape,Dense, Dropout, Embedding, LSTM,Flatten,Conv2D,MaxPooling2D,Bidirectional
from keras.optimizers import Adam,SGD

pos_oh={}
esize=300
sent_len=540
TRAINING=int(sys.argv[1])
class mine_model:
    def __init__(self,shapez):
        self.conv_1=Conv2D(96,(2,shapez),activation='relu')
        self.conv_2=Conv2D(96,(3,shapez),activation='relu')
        self.conv_3=Conv2D(96,(4,shapez),activation='relu')
        self.lstm1=Bidirectional(LSTM(96,return_sequences=True))
        self.lstm2=Bidirectional(LSTM(96,return_sequences=True))
        self.lstm3=Bidirectional(LSTM(96,return_sequences=True))
        self.lstm2_1=Bidirectional(LSTM(96))
        self.lstm2_2=Bidirectional(LSTM(96))
        self.lstm2_3=Bidirectional(LSTM(96))
        
        self.dense=Dense(256,activation='sigmoid')
    def forward(self,inp):
        conv1=self.conv_1(inp)
        conv2=self.conv_2(inp)
        conv3=self.conv_3(inp)
        pool1=MaxPooling2D((sent_len-1,1))(conv1)
        pool2=MaxPooling2D((sent_len-2,1))(conv2)
        pool3=MaxPooling2D((sent_len-3,1))(conv3)
        flat1=Flatten()(pool1)
        flat2=Flatten()(pool2)
        flat3=Flatten()(pool3)
        conc=Concatenate()([flat1,flat2,flat3])
        out=Dropout(0.25)(conc)
        dense=self.dense(out)
        dense=Dropout(0.5)(dense)
        return dense
word_layers=mine_model(esize)
pos_layers=mine_model(46)
def forward(inp,pos):
    wout=word_layers.forward(inp)
    pout=pos_layers.forward(pos)
    return (wout,pout)
utils.prepare_pos()
if(TRAINING):
    toffset=0
    foffset=0
    try:
        toffset=int(sys.argv[2])
        foffset=int(sys.argv[3])
    except:
        pass
    KTF.clear_session()
    left_inp=Input(shape=(sent_len,esize,1),dtype='float32')
    left_pos=Input(shape=(sent_len,46,1),dtype='float32')
    right_inp=Input(shape=(sent_len,esize,1),dtype='float32')
    right_pos=Input(shape=(sent_len,46,1),dtype='float32')
    left_out=forward(left_inp,left_pos)
    right_out=forward(right_inp,right_pos)
    #use left_out[0] and right_out[0] as input, left_out[1] and right_out[1] as output
    ana_in=Input(shape=(None,256),dtype='float32')
    old_model=load_model('mine.h5')
    print(old_model.summary())
    ana_hid=Dense(1024,activation='sigmoid')(ana_in)
    ana_out=Dense(256,activation='linear')(ana_hid)
    model=Model(inputs=ana_in,outputs=ana_out)
    model.compile(optimizer='adam',loss='MSE')
    utils.ana_train(old_model,model,TRAINING,2048,'ana','ana-test.txt',False,toffset,foffset)
else:
    min_loss=1000
    min_pos=-1
    for i in range(1,10):
        model=load_model('ana_%d.h5'%i)
        if(model):print('loaded %d'%i)
        ans=utils.mine_test(model,False)
        print(' '.join(ans))
        if(float(ans[3])<min_loss):
            min_loss=float(ans[3])
            min_pos=i
        os.system('cp ana_%d.h5 ana.h5'%min_pos)
    
