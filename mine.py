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
from keras.layers import Reshape,Dense, Dropout, Embedding, LSTM,Flatten,Conv1D,MaxPooling1D
from keras.optimizers import Adam

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        #self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # eij = K.dot(x, self.W) TF backend doesn't support it

        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
    #print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        #return input_shape[0], input_shape[-1]
        return input_shape[0],  self.features_dim
pos_oh={}
esize=300
sent_len=540
TRAINING=int(sys.argv[1])
class mine_model:
    def __init__(self):
        self.conv_1=Conv1D(128,2,padding='valid',activation='relu')
        self.conv_2=Conv1D(128,3,padding='valid',activation='relu')
        self.conv_3=Conv1D(128,4,padding='valid',activation='relu')
        self.lstm1=LSTM(96)
        self.lstm2=LSTM(96)
        self.lstm3=LSTM(96)
        self.dense=Dense(256,activation='relu')
    def forward(self,inp):
        inp=Reshape([sent_len,-1])(inp)
        conv_1=self.conv_1(inp)
        conv_2=self.conv_2(inp)
        conv_3=self.conv_3(inp)
        pool_1=MaxPooling1D(3)(conv_1)
        pool_2=MaxPooling1D(3)(conv_2)
        pool_3=MaxPooling1D(3)(conv_3)
        lstm1=self.lstm1(pool_1)
        lstm2=self.lstm2(pool_2)
        lstm3=self.lstm3(pool_3)
        out=Concatenate()([lstm1,lstm2,lstm3])
        out=Dropout(0.25)(out)
        dense=self.dense(out)
        dense=Dropout(0.5)(dense)
        return dense
word_layers=mine_model()
pos_layers=mine_model()
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
    if(os.path.isfile('mine-pr.h5')==False):
        model=Model(inputs=[left_inp,left_pos],outputs=predense)
        model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
        utils.mine_pretrain(model,TRAINING,2048,'mine',toffset)
    else:
        realdense=Dense(2,activation='softmax',name='realdense')(Concatenate()([left_out[0],left_out[1],right_out[0],right_out[1]]))
        model=Model(inputs=[left_inp,left_pos,right_inp,right_pos],outputs=realdense)
        model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
        model.load_weights('mine-pr.h5',by_name=True)
        utlis.mine_train(model,TRAINING,2048,'mine','mine-test.txt',False,toffset,foffset)
else:
    pass
    
