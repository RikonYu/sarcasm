import numpy
import re
import os
import sys
import gensim
import keras
import utils
from keras.models import Sequential, Model, load_model
from keras.layers import Input,Concatenate
from keras.layers import Reshape,Dense, Dropout, Embedding, LSTM,Flatten,Conv2D,MaxPooling2D
from keras.optimizers import Adam

TRAINING=int(sys.argv[1])
sent_len=540
esize=300
def get_out(inp):
    conv1=Conv2D(96,(2,esize),activation='relu')(inp)
    conv2=Conv2D(96,(3,esize),activation='relu')(inp)
    conv3=Conv2D(96,(4,esize),activation='relu')(inp)
    pool1=MaxPooling2D((sent_len-1,1))(conv1)
    pool2=MaxPooling2D((sent_len-2,1))(conv2)
    pool3=MaxPooling2D((sent_len-3,1))(conv3)
    flat1=Flatten()(pool1)
    flat2=Flatten()(pool2)
    flat3=Flatten()(pool3)
    conc=Concatenate([flat1,flat2,flat3])
    dense=Dense(256,activation='relu')(conc)
    return dense
if(TRAINING):
    toffset=0
    foffset=0
    try:
        toffset=int(sys.argv[2])
        foffset=int(sys.argv[3])
        
    except:
        pass
    inp=Input(shape=(sent_len*2,esize,1),dtype='float32')
    add_inp=Input(shape=(sent_len,esize,1),dtype='float32')
    pre_dense=get_out(inp)
    real_dense=get_out(add_inp)
    pre_out=Dense(2,activation='softmax')(pre_dense)
    real_out=Dense(2,activation='softmax')(Concatenate([pre_dense,real_dense]))
    model=Model(inputs=inp,outputs=pre_out)
    if(os.path.isfile('pretrain1-pr.h5')==False):
    #pre-train
        model.compile(optimizer='adam',loss='categorical_crossentropy')
        utils.pretrain(model,TRAINING,2048,'pretrain1_result0.txt',toffset)
    else:
    #train
        model.load_weights("pretrain1-pr.h5")
        model=Model(inputs=[inp,add_inp],outputs=real_out)
        model.compile(optimizer='adam',loss='categorical_crossentropy')
        model=utils.train(model,TRAINING,2048,'pretrain1','pretrain1-test.txt',False,toffset,foffset)
else:
    if(os.path.isfile('pretrain1-pr.h5')==False):
        min_loss=1000
        min_pos=-1
        for i in range(1,10):
            model=load_model('pretrain1_pr_%d.h5'%i)
            ans=utils.test(model,True)
            print(' '.join(ans))
            if(float(ans[3])<min_loss):
                min_loss=float(ans[3])
                min_pos=i
            os.system('cp pretrain1_pr_%d.h5 pretrain1-pr.h5'%min_pos)
    else:
        min_loss=1000
        min_pos=-1
        for i in range(1,10):
            model=load_model('pretrain1_%d.h5'%i)
            ans=utils.test(model,True)
            print(' '.join(ans))
            if(float(ans[3])<min_loss):
                min_loss=float(ans[3])
                min_pos=i
            os.system('cp pretrain1_%d.h5 pretrain1.h5'%min_pos)
