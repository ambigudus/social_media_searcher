# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 13:33:04 2019

@author: houwenxin
"""
import numpy as np
import pandas as pd
import jieba
from cluster import Clusters
from collections import Counter
import time
import lshforest
import datetime
from dateutil.relativedelta import relativedelta
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import DBSCAN
relativedelta(months=1)
data = pd.read_csv("./data/yule.csv",encoding='gbk',names=['time','content','forward','comment','like','url'])
data=data.dropna()
#print(data)
data['time']=pd.to_datetime(data['time'],format='%Y/%m/%d')
# data['month']=data['time'].map(lambda x: x.strftime('%Y-%m'))
# data['year']=data['time'].dt.year.to
# data['month']=data['time'].dt.month
# data['date_num']=data['time'].map(lambda x: x.year*12+x.month)
data['time']=pd.to_datetime(data['time'],format='%Y-%m-%d')
data['month']=data["time"].dt.month
data=data.sort_values(by='time',ascending=False)
dates =data['month'].tolist()
# data = lshforest.calres_small2(data)        #计算 innovation值
# len1 = len(data)
# data = data[data["innovation"] > data['innovation'].quantile(0.1)]
# len2 = len(data)
# print("筛选前的数据数：", len1)
# print("筛选后的数据数：", len2)
#dates =data['date_num'].tolist()
jieba.load_userdict('./data/jieba.txt') 
start=time.time()
stop_words = [word.strip() for word in open('./data/stop_words.txt', 'r',encoding='utf-8').readlines()]
# print(stop_words)
print("正在分词...")
# 分词，去除停用词
data['splitword']=data["content"].apply(lambda x: [word for word in jieba.cut(str(x)) if (word not in stop_words) and len(word)>1])
splitword = data['splitword'].tolist()

'''
to_sort = list(zip(dates, splitword))
to_sort = sorted(to_sort, key=lambda x:x[0], reverse=True)
dates[:], splitword[:] = zip(*to_sort)
'''

#合并每个时间窗内的keywords
time_word_dict = {}
count = {}
for i in range(len(dates)):
    if dates[i] in time_word_dict:
        time_word_dict[dates[i]] = time_word_dict[dates[i]] + splitword[i]
        count[dates[i]]+=1
    else:
        #print(dates[i])
        time_word_dict[dates[i]] = splitword[i]
        count[dates[i]]=1
count_times=[count[i] for i in sorted(count,reverse=True)]
#print(time_word_dict)
#print(len(time_word_dict))

#计算词频
for key,words in time_word_dict.items():
    word_count = {}
    for i in range(len(words)):
        word = words[i]
        if word in word_count:
            word_count[word]["F"] += 1
        else:
            word_count[word] = {"F":1, "R":0} # F：词频 Fi, R：增长率 Fi/(Fi-1 + 1)
    time_word_dict[key] = word_count

#计算词频增长率
for key,word_dict in time_word_dict.items():
    if key - 1 in time_word_dict:
        for word in word_dict:
            if word in time_word_dict[key-1]:
                time_word_dict[key][word]["R"] = time_word_dict[key][word]["F"] / (1 + time_word_dict[key-1][word]["F"])
            else:
                time_word_dict[key][word]["R"] = time_word_dict[key][word]["F"]

for key,word_dict in time_word_dict.items():
    maxF = 0.0
    maxR = 0.0
    for word, value_dict in word_dict.items():
        if value_dict["F"] > maxF:
            maxF = value_dict["F"]
        if value_dict["R"] > maxR:
            maxR = value_dict["R"]
    if maxF == 0 or maxR == 0:
        print("Something wrong with", key)
        continue
    for word, value_dict in word_dict.items():
        value_dict["F"] = value_dict["F"] / maxF
        value_dict["R"] = value_dict["R"] / maxR
        word_dict[word] = value_dict
    time_word_dict[key] = word_dict

F_output = []
R_output = []
output = []
for key,word_dict in time_word_dict.items():
    F_output.append(sorted(word_dict.items(), key=lambda x:x[1]["F"], reverse=True)[:250])  #筛选词频最大的120个
    R_output.append(sorted(word_dict.items(), key=lambda x:x[1]["R"], reverse=True)[:200])   #筛选词频变化率最大的80个
#print(len(F_output))
for i in range(len(F_output)):
    output.append([word for word in F_output[i] if word in R_output[i]])

dates = list(set(dates))
dates.sort(reverse=True)
print('共分成%d个时间段'%len(output))
kw=[]                       #每个时间窗的突发词字符串
for i in range(len(output)):
    kw.append(' '.join([output[i][j][0] for j in range(len(output[i]))]))

    print("月份:", dates[i],'\t%d个关键词'%len(output[i]))
    print("关键词:", end="")
    for j in range(len(output[i])):
        print(output[i][j][0], end=" ")
    print("\n")


#--------------聚类---------------
num=0
tags=[]
keywords_cluster_index=[]
keywords_cluster=[]
for i in range(len(output)): #按时间窗聚类
    word_list=[output[i][j][0] for j in range(len(output[i]))]  #每个时间窗的突发词list
    
    vecs=[]
    for k in range(num,num+count_times[i]):
        vec=np.array([word_list[j] in splitword[k] for j in range(len(word_list))])  #突发词是否在每个文本的分词
        vecs.append(vec.astype(int))
    clf=Clusters(vecs)              
    tag=clf.classify(len(set(tags)))
    tags.extend(tag)
    
    print('第%d月聚类结果):'%(dates[i]))
    result = Counter(tag)
    print (result.most_common(min(5,len(result))))
    
    vecs_kw=[]
    for j in range(len(word_list)):
        vec_kw=np.array([word_list[j] in splitword[m] for m in range(num,num+count_times[i])])
        vecs_kw.append(vec_kw.astype(int))
    #计算AP
    # ap = AffinityPropagation(preference=-26).fit(vecs_kw)
    # cluster_centers_indices = ap.cluster_centers_indices_    # 预测出的中心点的索引，如[123,23,34]
    # labels = ap.labels_    # 预测出的每个数据的类别标签,labels是一个NumPy数组
    # n_clusters_ = len(cluster_centers_indices) 
    # labels = ap.labels_ 
    # print(n_clusters_)
    # print(dates[i],word_list,'\n',labels,'\n')
    #labels = DBSCAN(eps = 0.8,).fit_predict(vecs_kw)
    clf=Clusters(vecs_kw)              
    labels=clf.classify(0)
    #tags.extend(tag)
    #print('第%d月聚类结果):'%(dates[i]))
    classify=[[word_list[x] for x in range(len(word_list)) if labels[x]==t] for t in range(len(set(labels)))]
    for p,string in enumerate(classify):
        classify[p]=' '.join(classify[p])
    keywords_cluster_index.extend([dates[i]]*len(classify))
    keywords_cluster.extend(classify)
    #print(classify)
    num+=count_times[i]
dff={"month": keywords_cluster_index,"keywords": keywords_cluster}
dfff = pd.DataFrame(dff,columns=['month', 'keywords']) 
dfff.to_csv('./output/aftercluster_keywords.csv',index=None) 


print('\n')
data.insert(0,'tag',tags)
data['heat']=data.apply(lambda x: x['forward']*0.5+x['comment']*0.4+x['like']*0.1,axis=1)   #计算热度
data['splitword_str']=data['splitword'].apply(lambda x: ' '.join(x)) 
df=data[['time','month','tag','heat','splitword_str','content']]
df=df.sort_values(by=['tag','month','heat'],ascending=[True,True,False])
df.to_csv('./output/total_result.csv',index=None)         #聚类结果

#得到聚类后的类关键词
cnt=0
tag_valid=[]
cluster_keyword=[]
max_heat_list=[]
content_list=[]
month_list=[]
time_list=[]
for i in range(len(set(tags))):
    if tags.count(i)>2: 
        tag_valid.append(i)
        cnt+=1    #集合元素 ＞3 当做一个有效聚类
        kwl=data[data.tag==i]['splitword'].tolist()
        month=data[data.tag==i]['month'].tolist()[0]
        index=dates.index(month)
        kwl=sum(kwl,[])
        output_=[output[index][i][0] for i in range(len(output[index]))]
        kwl=[word for word in kwl if word in output_]
        result=Counter(kwl)
        n=min(3,len(result))
        cluster_key=[result.most_common(n)[i][0] for i in range(n)]
        print('第%d个有效聚类(month=%d tag=%d)关键词为：'%(cnt,month,i),cluster_key)
        cluster_keyword.append(' '.join(cluster_key))
        time_list.append( df[df.tag==i].groupby('tag').apply(lambda x: x[x.heat==x.heat.max()]).iat[0,0] )
        max_heat_list.append( df[df.tag==i].groupby('tag').apply(lambda x: x[x.heat==x.heat.max()]).iat[0,3] )
        content_list.append( df[df.tag==i].groupby('tag').apply(lambda x: x[x.heat==x.heat.max()]).iat[0,5] )
        month_list.append(month)
tag_csv={'time':time_list, 'month':month_list,'tags':tag_valid,'cluster_keywords':cluster_keyword,'heat':max_heat_list,'content':content_list}
df_tag=pd.DataFrame(tag_csv,columns=['time','month','tags', 'cluster_keywords','heat','content'])
df_tag.to_csv('./output/cluster_keywords.csv',index=None)  #保存有效聚类的关键词

c={"month": dates,"keywords": kw}
kw = pd.DataFrame(c,columns=['month', 'keywords']) 
kw.to_csv('./output/period_keywords.csv',index=None)        #输出每个时间窗的关键词

data[['time','heat','content','splitword_str']].to_csv('./output/content&splitword.csv',index=None)

print(time.time()-start)
