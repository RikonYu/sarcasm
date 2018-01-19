import csv
import re
import numpy
import os
import random
import pickle
from collections import Counter
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

f=open("sarcasm_v2.csv",'r',encoding='utf-8')
fsize=os.path.getsize('sarcasm_v2.csv')
reader=csv.reader(f)
'''
bow=Counter()
for row in reader:
    bow=bow+Counter(re.compile("[,?;\"\' !.]").split(row[3]))
    bow=bow+Counter(re.compile("[,?;\"\' !.]").split(row[4]))
    
ks=list(bow.keys())
pickle.dump(ks,open("words.txt",'wb'))
'''

#cls=MLPClassifier(hidden_layer_sizes=(2),max_iter=500,activation='logistic')
#cls=GaussianNB()
cls=SVC()
ks=pickle.load(open("words.txt",'rb'))
rks=dict([(ks[i],i) for i in range(len(ks))])

def bow(sq,sa):
    ans=numpy.zeros([2*len(ks)])
    a=re.compile("[,?;\"\' !.]").split(sq)
    ba=Counter(a)
    for i in ba.keys():
        ans[rks[i]]=ba[i]
    a=re.compile("[,?;\"\' !.]").split(sa)
    ba=Counter(a)
    for i in ba.keys():
        ans[rks[i]+len(ks)]=ba[i]
    return ans

X=numpy.zeros([4693,59572])
Y=numpy.zeros([4693])

sord=numpy.random.choice(4693,4693,replace=False)

mk=0
for row in reader:
    X[mk]=bow(row[3],row[4])
    Y[mk]=(row[1] in 'sarc')
    mk+=1
'''
for k in range(batch_size):
    offset=random.randrange(fsize-20000)
    f.seek(offset)
    try:
        f.readline()
    except:
        break
    gets=f.readline()
    if(len(gets)==0):
        f.seek(0)
        f.readline()
        gets=f.readline()
    gets=list(csv.reader([gets]))
    X.append(bow(gets[0][3],gets[0][4]))

    Y.append(gets[0][1] in 'sarc')
    #print(gets[0][1],gets[0][1] in 'sarc')]
X.append(numpy.zeros([len(ks)*2]))
X.append(numpy.zeros([len(ks)*2]))
Y.append(True)
Y.append(False)
'''
cls.fit(X[sord[:4400]][:],Y[sord[:4400]])
print(sum(cls.predict(X[sord[4400:]][:])==numpy.array(Y[sord[4400:]]))/float(len(Y[4400:])))

offset=fsize-20000
f.seek(offset)
f.readline()
X=[]
Y=[]
