import ijson
import csv
#fids=open('wlist.txt','w')
f=open('main/comments.json','r')
data={}
x=[0 for i in range(3)]
ind=0
csvr=csv.reader(open('main/train-balanced.csv'),delimiter='|')
for row in csvr:
    #print(row)
    rs=row[1].split(' ')
    for i in row[0].split():
        data[i]=1
    data[rs[0]]=(row[0],(row[2][0]=='1'))
    data[rs[1]]=(row[0],(row[2][2]=='1'))
    #print((row[2][0]=='1'),(row[2][2]=='1'))
#raise Exception
for prefix, the_type, value in ijson.parse(f):
    if(ind==1):
        x[0]=value
    elif(ind==4):
        x[1]=value
    elif(ind==6):
        x[2]=value
    if(ind>=19):
        
        if(x[0]=='c07fd66'):
            print(x[0] in data)
        '''
        if(x[0] in data):
            #print(x)
            fids.write(x[0]+' |@| '+x[1]+' |@| '+str(x[2])+'\n')
        '''
            #ins.append([float(data[x[0]][1]),x[1]])            
        #X.append(x)
        ind=0
    ind+=1
