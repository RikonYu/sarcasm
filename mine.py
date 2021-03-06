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
        #lstm1=self.lstm1(conv_1)
        #lstm2=self.lstm2(conv_2)
        #lstm3=self.lstm3(conv_3)
        #lstm21=self.lstm2_1(lstm1)
        #lstm22=self.lstm2_2(lstm2)
        #lstm23=self.lstm2_3(lstm3)
        flat1=Flatten()(pool1)
        flat2=Flatten()(pool2)
        flat3=Flatten()(pool3)
        conc=Concatenate()([flat1,flat2,flat3])
        #out=Concatenate()([lstm21,lstm22,lstm23])
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
    predense=Dense(2,activation='softmax',name='predense')(Concatenate()([left_out[0],left_out[1]]))
    opt=SGD(lr=0.0003,momentum=0.9,decay=1e-6)
    if(os.path.isfile('mine-pr.h5')==False):
        model=Model(inputs=[left_inp,left_pos],outputs=predense)
        model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
        utils.mine_pretrain(model,TRAINING,2048,'mine',toffset)
    else:
        realdense=Dense(2,activation='softmax',name='realdense')(Concatenate()([left_out[0],left_out[1],right_out[0],right_out[1]]))
        model=Model(inputs=[left_inp,left_pos,right_inp,right_pos],outputs=realdense)
        model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        #if(os.path.isfile('mine_%d.h5'%TRAINING)==False and os.path.isfile('mine_%d.h5'%(TRAINING+1))==False and toffset==0):
        #    model.load_weights('mine-pr.h5',by_name=True)
        utils.mine_train(model,TRAINING,2048,'mine','mine-test.txt',False,toffset,foffset)
else:
    min_loss=1000
    min_pos=-1
    for i in range(1,10):
        model=load_model('mine_%d.h5'%i)
        if(model):print('loaded %d'%i)
        ans=utils.mine_test(model,False)
        print(' '.join(ans))
        if(float(ans[3])<min_loss):
            min_loss=float(ans[3])
            min_pos=i
        os.system('cp mine_%d.h5 mine.h5'%min_pos)
    
