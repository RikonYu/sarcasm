import numpy
import re
import utils
import gensim
import os,sys
from keras import backend as KTF
from keras.models import Sequential, Model, load_model
from keras.layers import Reshape,Dense, Dropout, Embedding, LSTM,Flatten,Conv2D,MaxPooling2D,Input, Conv1D
from keras.optimizers import Adam
TRAINING=int(sys.argv[1])
esize=300
sent_len=540
def forward(inp):
    real_inp=Reshape((sent_len*2,esize))(inp)
    conv1=Conv1D(256,3,activation='sigmoid',padding='valid',kernel_initializer='he_normal')(real_inp)
    conv2=Conv1D(256,3,activation='sigmoid',padding='valid',kernel_initializer='he_normal')(conv1)
    #conv2=Conv2D(256,(2,1),activation='sigmoid',padding='valid')(conv1)
    #conv2=Reshape((sent_len*2-2,256))(conv1)
    conv2=Dropout(0.25)(conv2)
    lstm1=LSTM(256,activation='sigmoid',kernel_initializer='he_normal',return_sequences=True,dropout=0.5)(conv2)
    lstm2=LSTM(256,activation='sigmoid',kernel_initializer='he_normal',dropout=0.5)(lstm1)

    #raise Exception
    dense=Dense(256,activation='sigmoid')(lstm2)
    out=Dense(2,activation='softmax')(dense)
    return out
if(TRAINING):
    KTF.clear_session()
    inp=Input(shape=(sent_len*2,esize,1),dtype='float32')
    out=forward(inp)
    model=Model(inputs=inp,outputs=out)
    toffset=0
    foffset=0
    if(len(sys.argv)>2):
        toffset=int(sys.argv[2])
        foffset=int(sys.argv[3])
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    utils.train(model,TRAINING,2048,'baseline2','baseline2-test.txt',True,toffset,foffset)
else:
    min_loss=1000
    min_pos=-1
    for i in range(1,9):
        model=load_model('baseline2_%d.h5'%i)
        ans=utils.test(model,True)
        print(' '.join(ans))
        if(float(ans[3])<min_loss):
            min_loss=float(ans[3])
            min_pos=i
    os.system('cp baseline2_%d.h5 baseline2.h5'%min_pos)
