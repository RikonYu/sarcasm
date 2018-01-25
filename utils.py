import keras
import numpy
import gensim
import re
import csv

embedding_model=gensim.models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin', binary=True)
esize=300
sent_len=0
def set_sentlen(l):
    global sent_len
    sent_len=l

def read_embedding(words,sent_len):
    X=[numpy.resize(numpy.array([embedding_model[word] for word in sent if word in embedding_model]),[sent_len,esize]) for sent in words]
    return X

def clean_up(s,sent_len):
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
            if(len(ins)>meta_batch):
                ins=clean_up(ins)
                history=model.fit(ins[0],ins[1],epochs=1,verbose=2,batch_size=bbatch,validation_split=0)
                fout.write(history.history['loss'][0])
                fout.write('\n')
                ins=[]
        fsent.seek(0)
        next(sentreader)
    return model

def train(model,max_epoch,batch_size,foutname,singular=True):
    ins=[]
    ftrue=open('true_context.csv','r')
    ffalse=open('false_context.csv','r')
    fout=open(foutname,'w')
    global sent_len
    treader=csv.reader(ftrue,delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
    freader=csv.reader(ffalse,delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
    for epoch in range(max_epoch):
        while(True):
            try:
                trues=next(treader)
                falses=next(freader)
            except:
                break
            ins.append([True,trues[0],trues[1]])
            ins.append([False,falses[0],falses[1]])
            if(len(ins)>=batch_size):
                ins=clean_up(ins,sent_len)
                if(singular==False):
                    history=model.fit([ins[0],ins[1]],ins[2],epochs=1,verbose=2,validation_split=0.05)
                else:
                    #print((ins[0]+ins[1]),ins[2])
                    #raise Exception
                    history=model.fit(ins[0]+ins[1],ins[2],epochs=1,verbose=2,validation_split=0.05)
                #print(ins)
                fout.write(str(history.history['loss'][0]))
                fout.write('\n')
                ins=[]
            ftrue.seek(0)
            ffalse.seek(0)
    return model

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
        ins=clean_up([[int(row[2]),row[0],row[1]]],sent_len)
        if(singular==True):
            ans=model.predict([a+b for a,b in zip(ins[0],ins[1])])
        else:
            ans=model.predict([ins[0],ins[1]])
        loss-=numpy.log(ans[0][1])
        correct+=int(numpy.argmax(ans,axis=1)[0]==int(row[2]))
        total+=1
    return 'accuracy:',correct/total,'CE loss:',loss/total
        
