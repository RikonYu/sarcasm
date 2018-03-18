import keras
import numpy
import gensim
import re
import pickle
import csv
import gc
import time
import subprocess
from keras.models import load_model
import os
import tensorflow as tf
from keras import backend as KTF
from keras.callbacks import EarlyStopping
import os
embedding_model=gensim.models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin', binary=True)
esize=300
os.environ["CUDA_VISIBLE_DEVICES"]='5'
sent_len=540

def read_embedding(words,sent_len):
    X=list(filter(lambda i: i in embedding_model.vocab,words))
    if(X==[]):
        return numpy.zeros([sent_len,esize])
    X=numpy.resize(numpy.array(embedding_model[X]),[sent_len,esize])
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
    ftrue=open('./true_context.csv','r')
    ffalse=open('./false_context.csv','r')
    ftest=open('./test_context.csv','r')
    treader=csv.reader(ftrue,delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
    freader=csv.reader(ffalse,delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
    testder=csv.reader(ftest,delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
    twriter=open('./true_pickled.txt','w')
    fwriter=open('./false_pickled.txt','w')
    tester=open('./test_pickled.txt','w')
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
    '''
    fwriter.close()
    while(True):
        try:
            row=next(testder)
        except:
            break
        ins=clean_up([int(row[2]),row[0],row[1]],sent_len)
        pickle.dump(ins,tester)
    tester.close()
    '''
            
def pretrain(default_model,epoch,batch_size,foutname,offset):
    start_time=time.time()
    model=None
    model_name=(foutname+'_pr_'+str(epoch)+'.h5')
    fsent=open('../Sentiment.csv','r',encoding='utf-8')
    fout=open('pre'+foutname,'w')
    global sent_len
    ins=[]
    if(offset==0):
        if(os.path.isfile(model_name)):
            model=load_model(model_name)
            print('found trained model %s,proceed'%model_name)
            subprocess.Popen(['python3',foutname+'.py',str(epoch-1),'0','0'])
            return
        elif(os.path.isfile((foutname+'_'+str(epoch+1)+'.h5'))):
            model=load_model((foutname+'_'+str(epoch+1)+'.h5'))
            print('found latest model %s, training'%(foutname+'_'+str(epoch+1)+'.h5'))
        else:
            model=default_model
            print('using blank model')
    else:
        model=load_model(model_name)
        print('resume %s'%model_name)
    fsent.seek(offset)
    if(offset==0):
        fsent.readline()
    while(True):
        rdr=fsent.readline()
        if(not rdr):
            break
        row=next(csv.reader([rdr],delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL))
        ins.append([int(row[1]),row[3]])
        if(len(ins)>batch_size):
            ins=clean_up(ins,sent_len)
            history=model.fit(ins[0],ins[1],epochs=1,verbose=0,validation_split=0)
            fout.write(str(history.history['loss'][0]))
            fout.write('\n')
            ins=[]
        if(time.time()-start_time>=1900):
            model.save(model_name)
            subprocess.Popen(['python3',foutname+'.py',str(epoch),str(fsent.tell()),0])
            return
    if(epoch>0):
        subprocess.Popen(['python3',foutname+'.py',str(epoch-1),str(fsent.tell()),0])
        return
        
def train(default_model,epoch,batch_size,foutname,testoutname,singular,toffset=0,foffset=0):
    start_time=time.time()
    model=None
    global sent_len
    model_name=(foutname+'_'+str(epoch)+'.h5')
    if(epoch<0):
        return
    ins=[]
    ftrue=open('./true_context.csv','r')
    ffalse=open('./false_context.csv','r')
    try:
        os.remove(testoutname)
        os.remove(foutname)
    except:
        pass

    if(toffset==0):
        if(os.path.isfile(model_name)):
            model=load_model(model_name)
            print('found trained model %s,proceed'%model_name)
            subprocess.Popen(['python3',foutname+'.py',str(epoch-1),'0','0'])
            return
        elif(os.path.isfile((foutname+'_'+str(epoch+1)+'.h5'))):
            model=load_model((foutname+'_'+str(epoch+1)+'.h5'))
            print('found latest model %s, training'%(foutname+'_'+str(epoch+1)+'.h5'))
        else:
            model=default_model
            print('using blank model')
    else:
        model=load_model(model_name)
        print('resume %s'%model_name)
        
    ftrue.seek(toffset)
    ffalse.seek(foffset)
    
    print("Start training on epoch %d"%epoch)
    #ftest=open(testoutname,'a')
    fout=open(foutname+'.txt','a')
    while(True):
        trues=ftrue.readline()
        falses=ffalse.readline()
        if(not trues):
            break
        if(not falses):
            break
        tr=csv.reader([trues],delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
        fr=csv.reader([falses],delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
        trues=next(tr)
        falses=next(fr)
            
        tins=[True,trues[0],trues[1]]
        fins=[False,falses[0],falses[1]]
        tins=clean_up(tins,sent_len)
        fins=clean_up(fins,sent_len)
        ins.append(tins)
        ins.append(fins)
        if(len(ins)>=batch_size and batch_size>0):
            if(singular==False):
                x0=numpy.stack([k[0] for k in ins])
                x1=numpy.stack([k[1] for k in ins])
                y=numpy.stack([k[2] for k in ins])
                history=model.fit([x0,x1],y,epochs=1,verbose=0,validation_split=0)
            else:
                x0=numpy.stack([k[0] for k in ins])
                x1=numpy.stack([k[1] for k in ins])
                y=numpy.stack([k[2] for k in ins])
                history=model.fit(numpy.concatenate((x0,x1),axis=1),y,epochs=1,verbose=2,validation_split=0)
            fout.write(str(history.history['loss'][0]))
            fout.write('\n')
            ins=[]
        if(time.time()-start_time>=1900):
            model.save(model_name)
            subprocess.Popen(['python3',foutname+'.py',str(epoch),str(ftrue.tell()),str(ffalse.tell())])
            return
            
    model.save(model_name)
    #ftest.write(('epoch:%d '%epoch)+' '.join(test(model,singular))+'\n')
    fout.close()
    ftrue.close()
    ffalse.close()
    #ftest.close()
    if(epoch>0):
        subprocess.Popen(['python3',foutname+'.py',str(epoch-1),'0','0'])
        return
    #return model
'''
def train(model,max_epoch,batch_size,foutname,testoutname,singular=True):
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
'''
def test(model,singular=True):
    total=0
    correct=0
    loss=0
    global sent_len
    ftest=open('test_context.csv','r')
    treader=csv.reader(ftest,delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
    while(True):
        try:
            row=next(treader)
        except:
            break
        ins=clean_up([int(row[2]),row[0],row[1]],sent_len)
        if(singular==True):
            ans=model.predict(numpy.concatenate(([ins[0]],[ins[1]]),axis=1))
        else:
            ans=model.predict([[ins[0]],[ins[1]]])
        loss-=numpy.log2(ans[0][1])
        correct+=int(numpy.argmax(ans,axis=1)[0]==int(row[2]))
        total+=1
    return 'accuracy:',str(correct/total),'CE loss:',str(loss/total)
        
if __name__=='__main__':
    ex=[]
    print(read_embedding(ex,200))
