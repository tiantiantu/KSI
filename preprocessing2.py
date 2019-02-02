import codecs
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


wikivocab={}
file1=codecs.open("wikipedia_knowledge",'r','utf-8')
line=file1.readline()
while line:
    if line[0:3]!='XXX':
        line=line.strip('\n')
        line=line.split()
        for i in line:
            wikivocab[i.lower()]=1
    line=file1.readline()




notesvocab={}
filec=codecs.open("combined_dataset",'r','utf-8')

line=filec.readline()

while line:
    line=line.strip('\n')
    line=line.split()
    
    if line[0]=='codes:':
        line=filec.readline()
        line=line.strip('\n')
        line=line.split()
        
        if  line[0]=='notes:':
            
            line=filec.readline()
            
            while line!='end!\n':
                line=line.strip('\n')
                line=line.split()
                for word in line:
                    notesvocab[word]=1
                
                line=filec.readline()
                
            
    line=filec.readline()



a1=set(notesvocab)
a2=set(wikivocab)
a3=a1.intersection(a2)


wikidocuments=[]
file2=codecs.open("wikipedia_knowledge",'r','utf-8')
line=file2.readline()
while line:
    if line[0:4]=='XXXd':
        tempf=[]
        line=file2.readline()
        while line[0:4]!='XXXe':
            line=line.strip('\n')
            line=line.split()
            for i in line:
                if i.lower() in a3:
                    tempf.append(i.lower())
            line=file2.readline()
        wikidocuments.append(tempf)
        
    line=file2.readline()


notesdocuments=[]
file3=codecs.open("combined_dataset",'r','utf-8')

line=file3.readline()

while line:
    line=line.strip('\n')
    line=line.split()
    if line[0]=='codes:':
        line=file3.readline()
        line=line.strip('\n')
        line=line.split()
        
        if  line[0]=='notes:':
            tempf=[]
            line=file3.readline()
        
            while line!='end!\n':
                line=line.strip('\n')
                line=line.split()
                for word in line:
                    if word in a3:
                        tempf.append(word)
                
                line=file3.readline()
                
            
            notesdocuments.append(tempf)
    line=file3.readline()

####################################################################################

notesvocab={}
for i in notesdocuments:
    for j in i:
        if j.lower() not in notesvocab:
            notesvocab[j.lower()]=len(notesvocab)
notedata=[]
for i in notesdocuments:
    temp=''
    for j in i:
        temp=temp+j+" "
    notedata.append(temp)
    

wikidata=[]
for i in wikidocuments:
    temp=''
    for j in i:
        temp=temp+j+" "
    wikidata.append(temp)    
##########################################################

vect = CountVectorizer(min_df=1,vocabulary=notesvocab,binary=True)
binaryn = vect.fit_transform(notedata)
binaryn=binaryn.A
binaryn=np.array(binaryn,dtype=float)

vect2 = CountVectorizer(min_df=1,vocabulary=notesvocab,binary=True)
binaryk = vect2.fit_transform(wikidata)
binaryk=binaryk.A
binaryk=np.array(binaryk,dtype=float)


np.save('notevec',binaryn)
np.save('wikivec',binaryk)
