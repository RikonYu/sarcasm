import keras
import numpy
import gensim
import re
import pickle
import csv
import gc
import time
import os
import tensorflow as tf
from keras import backend as KTF
from keras.callbacks import EarlyStopping
import os
embedding_model=gensim.models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin', binary=True)
esize=300
os.environ["CUDA_VISIBLE_DEVICES"]='5,6'
sent_len=540

def read_embedding(words,sent_len):
    X=numpy.resize(numpy.array([embedding_model[word] for word in words if word in embedding_model]),[sent_len,esize])
    return X

def clean_up(x,sent_len):
    
    if(len(x)==2):
        y_=[1-x[0],int(x[0])]
        x_=re.findall(r'[\w]+',x[1])
        return [numpy.array(read_embedding(x_,sent_len)).reshape((sent_len,esize,1)),y_]
    else:
        y_=[1-x[0],int(x[0])]
        x_1=re.findall(r'[\w]+',x[1])
        x_2=re.findall(r'[\w]+',x[2])
        return [numpy.array(read_embedding(x_1,sent_len)).reshape((sent_len,esize,1)),
               numpy.array(read_embedding(x_2,sent_len)).reshape((sent_len,esize,1)),
               numpy.array(y_)]

def maker():
    ftrue=open('true_context.csv','r')
    ffalse=open('false_context.csv','r')
    ftest=open('test_context.csv','r')
    treader=csv.reader(ftrue,delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
    freader=csv.reader(ffalse,delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
    testder=csv.reader(ftest,delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
    twriter=open('true_pickled.txt','wb')
    fwriter=open('false_pickled.txt','wb')
    tester=open('tesst_pickled.txt','wb')
    while(True):
        try:
            trues=next(treader)
            falses=next(freader)
        except:
            break
        tins=[True,trues[0],trues[1]]
        fins=[False,falses[0],falses[1]]
        tins=clean_up(tins,sent_len)
        fins=clean_up(fins,sent_len)
        pickle.dump(tins,twriter)
        pickle.dump(fins,fwriter)
    twriter.close()
    fwriter.close()
    while(True):
        try:
            row=next(testder)
        except:
            break
        ins=clean_up([int(row[2]),row[0],row[1]],sent_len)
        pickle.dump(ins,tester)
    tester.close()
            
def pretrain(model,max_epoch,batch_size,foutname):
    fsent=open('../Sentiment.csv','r',encoding='utf-8')
    fout=open(foutname,'w')
    global sent_len
    sentreader=csv.reader(fsent,delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
    ins=[]
    next(sentreader)
    for epoch in range(max_epoch):
        for row in sentreader:
            ins.append([int(row[1]),row[3]])
            if(len(ins)>batch_size):
                ins=clean_up(ins,sent_len)
                history=model.fit(ins[0],ins[1],epochs=1,verbose=2,validation_split=0)
                fout.write(str(history.history['loss'][0]))
                fout.write('\n')
                ins=[]
        fsent.seek(0)
        next(sentreader)
    return model

def train(model,max_epoch,batch_size,foutname,testoutname,singular=True):
    '''
    xconfig = tf.ConfigProto(intra_op_parallelism_threads=2, 
                        inter_op_parallelism_threads=2,
                        allow_soft_placement=True, 
                        device_count = {'CPU': 5})
    session = tf.Session(config=xconfig)
    KTF.set_session(session)
    '''
    ins=[]
    tt=time.clock()
    ftrue=open('true_pickled.txt','rb')
    ffalse=open('false_pickled.txt','rb')
    try:
        os.remove(testoutname)
        os.remove(foutname)
    except:
        pass
    global sent_len
    for epoch in range(max_epoch):
        ftest=open(testoutname,'a')
        fout=open(foutname,'a')
        while(True):
            try:
                trues=pickle.load(ftrue)
                falses=pickle.load(ffalse)
            except:
                break
            ins.append(trues)
            ins.append(ffalses)
            if(len(ins)>=batch_size and batch_size>0):
                if(singular==False):
                    x0=numpy.stack([k[0] for k in ins])
                    x1=numpy.stack([k[1] for k in ans])
                    y=numpy.stack([k[2] for k in ans])
                    history=model.fit([x0,x1],y,epochs=1,verbose=2,validation_split=0)
                else:
                    x0=numpy.stack([k[0] for k in ins])
                    x1=numpy.stack([k[1] for k in ans])
                    y=numpy.stack([k[2] for k in ans])
                    history=model.fit(numpy.concatenate((x0,x1),axis=1),x2,epochs=1,verbose=2,validation_split=0)
                fout.write(str(history.history['loss'][0]))
                fout.write('\n')
                ins=[]
        print(time.clock()-tt)
        model.save(foutname+str(epoch)+'.h5')
        ftest.write(('epoch:%d '%epoch)+' '.join(test(model,singular))+'\n')
        fout.close()
        ftrue.seek(0)
        ffalse.seek(0)
    return model

def test(model,singular=True):
    '''
    xconfig = tf.ConfigProto(intra_op_parallelism_threads=2, 
                        inter_op_parallelism_threads=2,
                        allow_soft_placement=True, 
                        device_count = {'CPU': 5})
    session = tf.Session(config=xconfig)
    KTF.set_session(session)
    '''
    total=0
    correct=0
    loss=0
    global sent_len
    ftest=open('test_pickled.txt','rb')
    treader=csv.reader(ftest,delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
    while(True):
        try:
            ins=pickle.load(ftest)
        except:
            break
        if(singular==True):
            ans=model.predict(ins[0]+ins[1])
        else:
            ans=model.predict([ins[0],ins[1]])
        loss-=numpy.log(ans[0][1])
        correct+=int(numpy.argmax(ans,axis=1)[0]==int(row[2]))
        total+=1
    return 'accuracy:',str(correct/total),'CE loss:',str(loss/total)
        
if __name__=='__main__':
    maker()
