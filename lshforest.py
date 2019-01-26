from datasketch import MinHashLSHForest, MinHash
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer 
import numpy as np
import datetime

def mylshforest(corpus):
    #print(len(corpus))
	forest = MinHashLSHForest(num_perm=32)
	score_res=[0]
	mh=[]
	for i in range(len(corpus)-1):
		doc=corpus[i]
		doc2=corpus[i+1]
		m=MinHash(num_perm=32)
		for d in doc:
			m.update(d.encode('utf8'))
		forest.add(str(i),m)
		forest.index()
		mh.append(m)
		
		m2=MinHash(num_perm=32)
		for d in doc2:
			m2.update(d.encode('utf8'))
		result = forest.query(m2, 10)
		score=0.0
		for j in range(len(result)):
			score=score+m2.jaccard(mh[int(result[j])])
		if(len(result)>0):
			score=score/len(result)
		score_res.append(score)
		i=i+1
	return score_res

def calres_small2(data):
    
	starttime = datetime.datetime.now()
	f2=open('.\\data\\lshres.csv', 'w+',encoding='utf-8',errors='ignore')
	corpus=data["content"].tolist()
	
	weight=[]
    
	sum_doc=len(corpus)
	sum_word=0
	for doc in corpus:
		sum_word+=len(doc)
	avg_word=(sum_word/sum_doc)
	for i in range(sum_doc):
		tmp=len(corpus[i])/avg_word
		weight.append(tmp)		
	
	fs=mylshforest(corpus)
	#print(fs)
	corpus2=list(reversed(corpus))
	bs=mylshforest(corpus2)
	bs2=list(reversed(bs))
	#print(bs2)
	final_res=[]
	output = []
	count=0
	for i in range(len(fs)):
		f=(bs2[i]/(fs[i]+1))*weight[count]
		final_res.append(f)
		count=count+1
	for x in final_res:
		x = float(x - np.min(final_res))/(np.max(final_res)- np.min(final_res))
		x=x*100
		output.append(x)
		f2.write('%.2f' %x)
		f2.write('\n')
		count=count+1
	f2.close()
	data.insert(6, "innovation", output)
	endtime = datetime.datetime.now()
	print( (endtime - starttime).seconds )
	return data