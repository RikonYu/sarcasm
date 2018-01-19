import numpy
import scipy
import re
import os
import matplotlib.pyplot as plt
import gensim
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Input,Concatenate
from keras.layers import Reshape,Dense, Dropout, Embedding, LSTM,Flatten,Conv2D,MaxPooling2D
from keras.optimizers import Adam
import csv

TRAINING=1
embedding_model=gensim.models.KeyedVectors.load_word2vec_format('E:/wordembedding/GoogleNews-vectors-negative300.bin', binary=True)
esize=300
sent_len=540
bbatch=128
meta_batch=256
max_rounds=0
pretrain_rounds=2

def read_embedding(words,sent_len):
    X=[numpy.resize(numpy.array([embedding_model[word] for word in sent if word in embedding_model]),[sent_len,esize]) for sent in words]
    return X

def clean_up(s):
    global sent_len
    if(len(s[0])==2):
        y_=[[1-x[0],int(x[0])] for x in s]
        x_=[re.findall(r'[\w]+',x[1]) for x in s]
        return numpy.array(read_embedding(x_,sent_len)).reshape((len(s),sent_len,esize,1)),y_
    else:
        y_=[[1-x[0],int(x[0])] for x in s]
        x_1=[re.findall(r'[\w]+',x[1]) for x in s]
        x_2=[re.findall(r'[\w]+',x[2]) for x in s]
        return (numpy.array(read_embedding(x_1,sent_len)).reshape((len(s),sent_len,esize,1)),
               numpy.array(read_embedding(x_2,sent_len)).reshape((len(s),sent_len,esize,1)),
               y_)




if(TRAINING):
    model_input=Input(shape=(sent_len,esize,1),dtype='float32')
    add_input=Input(shape=(sent_len,esize,1),dtype='float32')
    conv_layer_1=Conv2D(256,(3,esize),activation='relu')
    conv_layer_2=Conv2D(256,(3,256),activation='relu')
    lstm_1=LSTM(256,activation='sigmoid',return_sequences=True)
    lstm_2=LSTM(256,activation='sigmoid')
    dropout=Dropout(0.5)
    dense_1=Dense(256,activation='sigmoid')
    left=conv_layer_1(model_input)
    left=Reshape((sent_len-2,256,1))(left)
    left=conv_layer_2(left)
    left=Reshape((sent_len-4,256))(left)
    left=lstm_1(left)
    left=lstm_2(left)
    left=dropout(left)
    left=dense_1(left)
    pretrain_out=Dense(2,activation='softmax')(left)

    right=conv_layer_1(add_input)
    right=Reshape((sent_len-2,256,1))(right)
    right=conv_layer_2(right)
    right=Reshape((sent_len-4,256))(right)
    right=lstm_1(right)
    right=lstm_2(right)
    right=dropout(right)
    right=dense_1(right)
    right=keras.layers.concatenate([left,right])
    real_out=Dense(2,activation='softmax')(right)

    if(os.path.exists('pretrain-deep-pre.h5')==False):
        model=Model(inputs=model_input,outputs=pretrain_out)
        model.compile(optimizer='rmsprop',loss='categorical_crossentropy')
        #dense_layer=Dense(256,activation='relu')
        

        #pre-train
        plt.ion()
        trds=0
        lloss=0.7
        disloss=0.7
        fsent=open('Sentiment.csv','r',encoding='utf-8')
        sentreader=csv.reader(fsent,delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
        ins=[]
        next(sentreader)
        for epoch in range(pretrain_rounds):
            for row in sentreader:
                ins.append([int(row[1]),row[3]])
                if(len(ins)>meta_batch):
                    #print("pretending pretraining")
                    ins=clean_up(ins)
                    history=model.fit(ins[0],ins[1],epochs=1,verbose=2,batch_size=bbatch,validation_split=0)
                    disloss+=history.history['loss'][0]
                    if(trds%20==19):
                        disloss/=20
                        plt.plot([trds//20,trds//20+1],[lloss,disloss],'b')
                        lloss=disloss
                        disloss=0
                    plt.pause(0.05)
                    trds+=1
                    ins=[]
            fsent.seek(0)
            next(sentreader)
        plt.show()
        model.save('pretrain-deep-pre.h5')
    else:
        #train
        model=load_model('pretrain-deep-pre.h5')
        model=Model(inputs=[model_input,add_input],outputs=real_out)
        model.compile(optimizer='adam',loss='categorical_crossentropy')

        ins=[]
        test_x=[]
        test_y=[]
        ftrue=open("true_context.csv",'r')
        ffalse=open("false_context.csv",'r')
        treader=csv.reader(ftrue,delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
        freader=csv.reader(ffalse,delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)

        ins=[]
        trials=0
        next(treader)
        next(freader)
        for i in range(bbatch//2):
            test_x.append(next(treader))
            test_x.append(next(freader))
            test_y.append(True)
            test_y.append(False)

        tests=[[test_y[i],test_x[i][0],test_x[i][1]] for i in range(len(test_x))]
        tests=clean_up(tests)

        tests_x_1=numpy.array(tests[0]).reshape([bbatch,sent_len,esize,1])
        tests_x_2=numpy.array(tests[1]).reshape([bbatch,sent_len,esize,1])
        tests_y=numpy.array(tests[2])
        trds=0
        lloss=0.7
        lacc=0.5
        lvacc=0.5
        lvloss=0.7
        for epoch in range(max_rounds):
            while(True):
                try:
                    trues=next(treader)
                    falses=next(freader)
                except:
                    break
                ins.append([True,trues[0],trues[1]])
                ins.append([False,falses[0],falses[1]])
                if(len(ins)>=meta_batch):
                    ins=clean_up(ins)
                    #print("pretending training")
                    history=model.fit([ins[0],ins[1]],ins[2],epochs=1,verbose=2,batch_size=bbatch,validation_split=0)
                    plt.plot([trds,trds+1],[lloss,history.history['loss'][0]],'r')
                    lloss=history.history['loss'][0]
                    plt.pause(0.05)
                    ins=[]
                    trds+=1
            ftrue.seek(0)
            ffalse.seek(0)
            next(treader)
            next(freader)
            for i in range(bbatch//2):
               next(treader)
               next(freader)
        plt.show()
        plt.savefig('pretrain-deep.jpg')
        model.save('pretrain-deep.h5')
else:
    total=0
    correct=0
    model=load_model('pretrain-deep.h5')
    ftest=open("test_context.csv",'r')
    treader=csv.reader(ftest,delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
    while(True):
        try:
            row=next(treader)
        except:
            break
        ins=clean_up([[int(row[2]),row[0],row[1]]])
        ans=model.predict([ins[0],ins[1]])
        correct+=int(numpy.argmax(ans,axis=1)[0]==int(row[2]))
        total+=1

        #raise Exception

    print(float(correc)/float(total))
        #raise Exception
