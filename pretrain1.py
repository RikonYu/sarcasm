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
utils.set_sentlen(sent_len)
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
    '''
if(TRAINING):
    model_input=Input(shape=(sent_len,esize,1),dtype='float32')
    add_input=Input(shape=(sent_len,esize,1),dtype='float32')
    conv_layer=Conv2D(256,(3,esize),activation='relu')
    dense_layer=Dense(256,activation='relu')
    conv1=conv_layer(model_input)
    conv2=conv_layer(add_input)
    pool1=MaxPooling2D((sent_len-2,1))(conv1)
    pool2=MaxPooling2D((sent_len-2,1))(conv2)

    flat1=Flatten()(pool1)
    flat2=Flatten()(pool2)

    dense1=dense_layer(flat1)
    dense2=dense_layer(flat2)
    pretrain_out=Dense(2,activation='softmax')(dense1)
    real_conc=keras.layers.concatenate([dense1,dense2])
    real_out=Dense(2,activation='softmax')(real_conc)
    '''
    toffset=0
    foffset=0
    try:
        toffset=int(sys.argv[2])
        foffset=int(sys.argv[2])
        
    except:
        pass
    inp=Input(shape=(sent_len*2,esize,1),dtype='float32')
    add_inp=Input(shape=(sent_len,esize,1),dtype='float32')
    pre_dense=get_out(inp)
    real_dense=get_out(add_inp)
    pre_out=Dense(2,activation='softmax')(pre_dense)
    real_out=Dense(2,activation='softmax')(Concatenate([pre_dense,real_dense]))
    model=Model(inputs=inp,outputs=pre_out)
    #pre-train
    if(os.path.isfile('pretrain-pr.h5')==False):
        
        model.compile(optimizer='adam',loss='categorical_crossentropy')
        utils.pretrain(model,5,1024,'pretrain1_result0.txt')
    model.load_weights("pretrain-pr.h5")
    #train
    model=Model(inputs=[inp,add_inp],outputs=real_out)
    model.compile(optimizer='adam',loss='categorical_crossentropy')
    model=utils.train(model,10,1024,'pretrain1','pretrain1-test.txt',False,0,0)
    model=Model(inputs=model_input,outputs=pretrain_out)
    model.compile(optimizer='adam',loss='categorical_crossentropy')
    
    #train
    model=Model(inputs=[model_input,add_input],outputs=real_out)
    model.compile(optimizer='adam',loss='categorical_crossentropy')
    model=utils.train(model,20,512,'pretrain1_result1.txt','pretrain1-test.txt',singular=False)
    model.save('pretrain_shallow.h5')
else:
    model=load_model('pretrain-shallow.h5')
    print(' '.join(utils.test(model,False)))
