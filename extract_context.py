import ijson
import csv
#fids=open('wlist.txt','w')
f=open('main/comments.json','r')
data={}
x=[0 for i in range(3)]
ind=0
csvr=csv.reader(open('main/train-balanced.csv',),delimiter='|')
for row in csvr:
    rs=row[1].split(' ')
    comid=''
    if(' ' in row[0]):
        comid=row[0].split()[1]
    data[rs[0]]=(comid,(row[2][0]=='1'))
    data[rs[1]]=(comid,(row[2][2]=='1'))
    #print((row[2][0]=='1'),(row[2][2]=='1'))

fin=open('wlist2.txt','r')
ftrue=open("./true_context.csv",'w',newline='')
ffalse=open("./false_context.csv",'w',newline='')
truewriter=csv.writer(ftrue,delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
falsewriter=csv.writer(ffalse,delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)

accr=fin.readlines()
accr=[i.split(' |@| ') for i in accr]
d={}
for i in accr:
    #print(i)
    d[i[1]]=(i[1],i[2].strip('\n'))
#print(d.keys())
for i in data.keys():
    if(data[i][0]==''):
        if(data[i][1]==True):
            truewriter.writerow(['',d[i][1]])
        else:
            falsewriter.writerow(['',d[i][1]])
    else:
        if(data[i][1]==True):
            truewriter.writerow([d[data[i][0]][1],d[i][1]])
        else:
            falsewriter.writerow([d[data[i][0]][1],d[i][1]])
        #raise Exception
