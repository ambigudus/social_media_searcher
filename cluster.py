import pandas as pd 
import numpy as np
import random

'''
with open('突发词',encoding='utf-8') as f:
    lines=f.readlines()
    words=[]
    for line in lines:
        words.append(words)

words=['慢性粒细胞白血病','金融科技','大数据']
def f(x):
    return x[4:].split()
text=pd.read_excel('data_out.xls')
text['关键词']=text['关键词'].apply(f)
keywords=text['关键词'].tolist()
vecs=[]
for i in range(len(keywords)):
    vec=np.array([words[j] in keywords[i] for j in range(len(words))])
    vecs.append(vec.astype(int))
#print(vecs[:4])
'''
class Clusters():
    def __init__(self, vecs):
        self.ls=list(range(len(vecs)))
        #self.len=len(self.ls)
        self.vecs=vecs
        self.clust=[]
        self.clust_zero=[]
        self.score=0
    def compare(self,v1,v2):
        sum1=v1.sum()
        sum2=v2.sum()
        #print(v1,sum1,v2,sum2)
        if v1.dot(v2)==0 :
            return False        
        scor=(min(sum1,sum2)-v1.dot(v2))/min(sum1,sum2)
        self.score+=scor
        #print(scor)       
        if sum1<=2 or sum2<=2:
            return scor ==0
        else:
            return scor< 0.6
    def cluster_compare(self,cluster,v,total,k):
        if total==k+1:
            return self.compare(self.vecs[cluster[k]],v)            
        else:
            if self.compare(self.vecs[cluster[k]],v):  
                return self.cluster_compare(cluster,v,total,k+1)
            else:
                return False
    def classify(self,tags_num):
        i1,i2=0,1#np.random.choice(self.ls,size=2,replace=False)
        TF=self.compare(self.vecs[i1],self.vecs[i2])
        if TF: 
            self.clust.append([i1, i2])
        else:
            self.clust.extend([[i1],[i2]])
        self.ls.remove(i1)
        self.ls.remove(i2)
        self.score=0
        
        while self.ls:
            scor=float('inf')
            i=self.ls[0]#random.choice(self.ls)
            if self.vecs[i].sum()==0:
                self.clust_zero.extend([[i]])
                self.ls.remove(i)
                continue
            out=-1
            #print(len(self.clust))
            for j in range(len(self.clust)):
                if self.cluster_compare(self.clust[j],self.vecs[i],len(self.clust[j]),0):
                    if self.score<scor:
                        scor=self.score
                        out=j
                self.score=0
            if out==-1:
                self.clust.extend([[i]])
            else:
                self.clust[out].append(i)
              
            self.ls.remove(i)
        self.clust+=self.clust_zero
        output=[0]*len(self.vecs)
        for i in range(len(self.clust)):
            for index in self.clust[i]:
                output[index]=i+tags_num
        return output
        #text['类别']=output
        #text.to_csv('text_cluster.csv')

#cls=Clusters(vecs)
#cls.classify()




