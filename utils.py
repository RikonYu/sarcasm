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

def getline(fin,offset):
    fin.seek(offset)
    k=fin.readline()
    if(not k):
        return -1
    row=next(csv.reader([k],delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL))
    return row

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
    ftest=open('../Sentiment.csv','r',encoding='utf-8')
    truepos=[0]
    falsepos=[0]
    testpos=[0]
    while(True):
        k=ftrue.readline()
        if(not k):
            break
        truepos.append(ftrue.tell())
    while(True):
        k=ffalse.readline()
        if(not k):
            break
        falsepos.append(ffalse.tell())
    while(True):
        k=ftest.readline()
        if(not k):
            break
        testpos.append(ftest.tell())
    pickle.dump(truepos,open('truepos.txt','wb'))
    pickle.dump(falsepos,open('falsepos.txt','wb'))
    pickle.dump(testpos,open('prepos.txt','wb'))
def pretrain(default_model,epoch,batch_size,foutname,offset,double=False):
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
        elif(os.path.isfile((foutname+'_pr_'+str(epoch+1)+'.h5'))):
            model=load_model((foutname+'_pr_'+str(epoch+1)+'.h5'))
            print('found latest model %s, training'%(foutname+'_'+str(epoch+1)+'.h5'))
        else:
            model=default_model
            print('using blank model')
    else:
        model=load_model(model_name)
        print('resume %s'%model_name)
    #fsent.seek(offset)
    posfile=open('prepos.txt','rb')
    prepos=pickle.load(posfile)[1:]
    posfile.close()
    if(offset==0):
        numpy.random.shuffle(prepos)
    for i in range(offset,len(prepos)):
        row=getline(fsent,prepos[i])
        #print(row)
        ins.append(clean_up([int(row[1]),row[3]],sent_len*(1+int(double))))
        if(len(ins)>=batch_size):
            x=numpy.stack([k[0] for k in ins])
            y=numpy.stack([k[1] for k in ins])
            history=model.fit(x,y,epochs=1,verbose=2,validation_split=0)
            fout.write(str(history.history['loss'][0]))
            fout.write('\n')
            ins=[]
        if(time.time()-start_time>=1900):
            model.save(model_name,overwrite=True)
            subprocess.Popen(['python3',foutname+'.py',str(epoch),str(fsent.tell()),'0'])
            return
    if(epoch>0):
        subprocess.Popen(['python3',foutname+'.py',str(epoch-1),'0','0'])
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
    tplaces=pickle.load(open('./truepos.txt','rb'))
    fplaces=pickle.load(open('./falsepos.txt','rb'))
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
    tposfile=open('truepos.txt','rb')
    fposfile=open('falsepos.txt','rb')
    tpos=pickle.load(tposfile)[:-1]
    fpos=pickle.load(fposfile)[:-1]
    tposfile.close()
    fposfile.close()
    if(toffset==0):
        #tposfile=open('truepos.txt','wb')
        #fposfile=open('falsepos.txt','wb')
        numpy.random.shuffle(tpos)
        numpy.random.shuffle(fpos)
        #pickle.dump(tpos,tposfile)
        #pickle.dump(fpos,fposfile)
        #tposfile.close()
        #fposfile.close()
    print("Start training on epoch %d"%epoch)
    fout=open(foutname+'.txt','a')
    for i in range(toffset,len(tpos)):
        '''
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
        '''
        trues=getline(ftrue,tpos[i])
        falses=getline(ffalse,fpos[i])
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
                history=model.fit([x0,x1],y,epochs=1,verbose=2,validation_split=0)
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
            subprocess.Popen(['python3',foutname+'.py',str(epoch),str(i),str(i)])
            return
            
    model.save(model_name)

    fout.close()
    ftrue.close()
    ffalse.close()
    if(epoch>0):
        subprocess.Popen(['python3',foutname+'.py',str(epoch-1),'0','0'])
        return
    
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
        #print(int(row[2]),numpy.concatenate(([ins[0]],[ins[1]]),axis=1).shape,numpy.array([1-int(row[2]),int(row[2])]).shape)
        if(singular==True):
            ans=model.evaluate(numpy.concatenate(([ins[0]],[ins[1]]),axis=1),numpy.array([1-int(row[2]),int(row[2])]))
        else:
            ans=model.evaluate(numpy.array([ins[0],ins[1]]),1)
        loss+=ans[0]
        correct+=ans[1]
        total+=1
    return 'accuracy:',str(correct/total),'CE loss:',str(loss/total)
if __name__=='__main__':
    maker()
    #pass
