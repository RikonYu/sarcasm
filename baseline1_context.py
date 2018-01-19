

import tensorflow as tf
import numpy
import scipy
import re
import csv
import matplotlib.pyplot as plt
import gensim
from tensorflow.python.framework import ops


ops.reset_default_graph()

embedding_model=gensim.models.KeyedVectors.load_word2vec_format('E:/wordembedding/GoogleNews-vectors-negative300.bin', binary=True)
esize=300
sent_len=700
bbatch=128

TRAINING=0



records=open('records.txt','w')

def read_embedding(words,sent_len):
    #X=[numpy.array([embedding_model[word] for word in sent if word in embedding_model]) for sent in words]
    #[i.resize([sent_len,esize],refcheck=False) for i in X]
    X=[numpy.resize(numpy.array([embedding_model[word] for word in sent if word in embedding_model]),[sent_len,esize]) for sent in words]
    #print(numpy.array(X).shape)
    return X

#print(read_embedding([['there','is','no','cow','level'],['black','sheep','wall']],))
#raise Exception
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


sess=tf.Session()

def get_weight(shape):
    return tf.Variable(tf.truncated_normal(shape=shape,mean=0.1,stddev=0.1))
def get_bias(shape):
    return tf.Variable(tf.constant(0.0,shape=shape))
def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='VALID')
def max_pool(x,height):
    return tf.nn.max_pool(x,strides=[1,1,1,1],ksize=[1,height,1,1],padding='VALID')
class NN:
    batch_size=10
    X=[]
    Y=[]
    h=[]
    pools=[]
    height=0
    cW=0
    cB=0
    dW1=0
    dB1=0
    dW2=0
    dB2=0
    ans=0
    loss=0
    train=0
    accuracy=0
    def __init__(self,batch_size,chsize,hidden_num,sentlen,old_NN=None):
        self.batch_size=batch_size
        self.X=tf.placeholder(tf.float32,[batch_size,None,esize,1])
        self.Y=tf.placeholder(tf.float32,[batch_size,2])
        self.height=sentlen
        
        if(old_NN!=None):
            self.cW=tf.identity(old_NN.cW)
            self.cB=old_NN.cB
            self.dW1=old_NN.dW1
            self.dB1=old_NN.dB1
            self.dW2=old_NN.dW2
            self.dB2=old_NN.dB2
        else:
            self.cW=get_weight([3,esize,1,chsize])
            self.cB=get_bias([chsize])
            self.dW1=get_weight([chsize,hidden_num])
            self.dB1=get_bias([hidden_num])
            #self.dW1=get_weight([chsize,2])
            #self.dB1=get_bias([2])
            
            self.dW2=get_weight([hidden_num,2])
            self.dB2=get_bias([2])
        self.h=tf.nn.relu(conv2d(self.X,self.cW)+self.cB)       
        self.pools=max_pool(self.h,self.height-2)
        self.ans=tf.reshape(self.pools,[self.batch_size,-1])
        self.ans=tf.matmul(self.ans,self.dW1)+self.dB1
        self.ans=tf.nn.sigmoid(self.ans)
        self.ans=tf.matmul(self.ans,self.dW2)+self.dB2
        self.loss=tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.Y,logits=self.ans))
        #self.loss=tf.reduce_mean(tf.nn.l2_loss(self.Y,self.ans))
        self.train_step=tf.train.AdamOptimizer(learning_rate=1e-2).minimize(self.loss)
        self.accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.ans,1),tf.argmax(self.Y,1)),tf.float32))
    def feed(self,x):
        return self.h.eval(session=sess,feed_dict={self.X:x})
    def test(self,X_,Y_):
        #print(X_)
        X_=numpy.array(X_)
        Y_=numpy.array(Y_)
        X_=numpy.reshape(X_,[X_.shape[0],X_.shape[1],X_.shape[2],1])
        return sess.run(self.accuracy,feed_dict={self.X:X_,self.Y:Y_})
        #return sess.run([tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y_,1),tf.argmax(self.Y,1)),tf.float32)),self.loss],feed_dict={self.X:X_,self.Y:Y_})
    def train(self,X_,Y_):
        X_=numpy.reshape(X_,[X_.shape[0],X_.shape[1],X_.shape[2],1])
        (_,lol,ans)=sess.run([self.train_step,self.loss,self.ans],feed_dict={self.X:X_,self.Y:Y_})
        #print(ans)
        ws=self.cW.eval(session=sess)
        #print('kernel CV:',numpy.var(ws)/numpy.average(ws))
        #print('ans CV:',numpy.var(ans)/numpy.average(ans))
        records.write(str(lol)+str(ans)+'\n')
        return lol
        
        
            



le=0


ins=[]


plt.ion()
last=-1
tlast=-1
alast=-1
dis=0
tdis=0
adis=0
epoch=0
k=NN(bbatch,96,512,sent_len)



#sess.run(tf.global_variables_initializer())
#tf.reset_default_graph()
tf_saver=tf.train.Saver()
tf_saver.restore(sess,'./baseline-cue-cnn-cont.ckpt')
if(TRAINING):
    test_x=[]
    test_y=[]
    ftrue=open("true_context.txt",'r')
    ffalse=open("false_context.txt",'r')

    ins=[]
    trials=0
    for i in range(bbatch//2):
        test_x.append(ftrue.readline())
        test_x.append(ffalse.readline())
        test_y.append(True)
        test_y.append(False)
    tests=[[test_y[i],test_x[i]] for i in range(len(test_x))]
    tests=clean_up(tests)
    for epoch in range(5):
        print('epoch:',epoch)
        ftrue.seek(0)
        ffalse.seek(0)
        for i in range(bbatch//2):
            ftrue.readline()
            ffalse.readline()
        while(True):
            trues=ftrue.readline()
            falses=ffalse.readline()
            if(not trues):
                break
            if(not falses):
                break
            ins.append([True,trues])
            ins.append([False,falses])
            if(len(ins)>=bbatch):
                ins=clean_up(ins)
                ttt=k.train(numpy.array(ins[0]),numpy.array(ins[1]))
                tttt,aaaa=k.test(tests[0],tests[1])
                tdis+=tttt
                dis+=ttt
                adis+=aaaa
                trials+=1
                #print(ttt)
                records.write(str(ttt)+'\n')
                if(trials%20==0):
                    dis/=20
                    tdis/=20
                    adis/=20
                    if(last==-1):
                        last=dis
                        tlast=tdis
                        alast=adis
                    plt.plot([trials/20,trials/20+1],[last,dis],'b')
                    plt.plot([trials/20,trials/20+1],[tlast,tdis],'r')
                    plt.plot([trials/20,trials/20+1],[alast,adis],'g')
                    
                    plt.pause(0.05)
                    last=dis
                    dis=0
                    tlast=tdis
                    tdis=0
                    alast=adis
                    adis=0
                    
                epoch+=1
                ins=[]
            #records.write('\n\n\n')

        tf_saver.save(sess,'./baseline-cue-cnn-cont.ckpt')
    #print(k.test(test_x,test_y))
    plt.savefig('baseline2_2.png')
else:
    correct=0
    total=0
    ftest=open("test_context.csv",'r')
    treader=csv.reader(ftest,delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
    ins=[]
    while(True):
        try:
            row=next(treader)
        except:
            break
        ins.append([int(row[2]),row[0]+row[1]])
        if(len(ins)==bbatch):
            ins=clean_up(ins)
            ans=k.test(ins[0],ins[1])
            #print(ans)
            correct+=ans
            total+=1
            ins=[]
    print(correct/float(total))
