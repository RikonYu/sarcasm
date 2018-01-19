import numpy
import re
import utils
import gensim
from keras.models import Sequential, Model, load_model
from keras.layers import Reshape,Dense, Dropout, Embedding, LSTM,Flatten,Conv2D,MaxPooling2D
from keras.optimizers import Adam
TRAINING=1
esize=300
sent_len=700
utils.set_sentlen(sent_len)
if(TRAINING):
    model=Sequential()
    model.add(Conv2D(256,(3,esize),activation='relu',padding='valid',input_shape=(sent_len,esize,1)))
    model.add(Reshape((sent_len-2,256,1)))
    model.add(Conv2D(256,(3,256),activation='relu',padding='valid'))
    model.add(Reshape((sent_len-4,256)))
    model.add(LSTM(256,activation='sigmoid',return_sequences=True))
    model.add(LSTM(256,activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(256,activation='sigmoid'))
    model.add(Dense(2,activation='softmax'))
    cadam=Adam(lr=0.01)
    model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
    model=utils.train(model,10,512,'baseline2.txt',singular=True)
    model.save('baseline2.h5')
else:
    model=load_model('baseline2.h5')
    print(' '.join(utils.test(model,True)))
