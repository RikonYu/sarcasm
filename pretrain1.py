import numpy
import re
import gensim
import keras
import utils
from keras.models import Sequential, Model, load_model
from keras.layers import Input,Concatenate
from keras.layers import Reshape,Dense, Dropout, Embedding, LSTM,Flatten,Conv2D,MaxPooling2D
from keras.optimizers import Adam

TRAINING=1
sent_len=540
esize=300
utils.set_sentlen(sent_len)
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
    model=Model(inputs=model_input,outputs=pretrain_out)
    model.compile(optimizer='adam',loss='categorical_crossentropy')


    #pre-train
    model=utils.pretrain(model,5,512,'pretrain1_result0.txt')
    model.save('pretrain-shallow-5.h5')
    #train
    model=Model(inputs=[model_input,add_input],outputs=real_out)
    model.compile(optimizer='adam',loss='categorical_crossentropy')
    model=utils.train(model,20,512,'pretrain1_result1.txt','pretrain1-test.txt',singular=False)
    model.save('pretrain_shallow.h5')
else:
    model=load_model('pretrain-shallow.h5')
    print(' '.join(utils.test(model,False)))
