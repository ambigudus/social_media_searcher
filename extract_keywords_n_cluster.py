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

data = pd.read_csv("./data/gqt.csv",encoding='gbk',names=['time','content','forward','comment','like','url'])
data=data.dropna()
#print(data)
data['time']=pd.to_datetime(data['time'],format='%Y-%m-%d')
data['month']=data["time"].dt.month
data=data.sort_values(by='time',ascending=False)
dates =data['month'].tolist()


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
    F_output.append(sorted(word_dict.items(), key=lambda x:x[1]["F"], reverse=True)[:120])  #筛选词频最大的120个
    R_output.append(sorted(word_dict.items(), key=lambda x:x[1]["R"], reverse=True)[:80])   #筛选词频变化率最大的80个
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
for i in range(len(output)): #按时间窗聚类
    word_list=[output[i][j][0] for j in range(len(output[i]))]  #每个时间窗的突发词list

    vecs=[]

    for k in range(num,num+count_times[i]):
        vec=np.array([word_list[j] in splitword[k] for j in range(len(word_list))])  #突发词是否在每个文本的分词
        vecs.append(vec.astype(int))
    clf=Clusters(vecs)              
    tag=clf.classify(len(set(tags)))

    tags.extend(tag)
    num+=count_times[i]
    print('第%d月聚成了%d类(大于1的集合为1个类):'%(dates[i],count_times[i]-len(set(tag))))
    result = Counter(tag)
    print (result.most_common(min(5,len(result))))

print('\n')
data.insert(0,'tag',tags)
data['heat']=data.apply(lambda x: x['forward']*0.5+x['comment']*0.4+x['like']*0.1,axis=1)   #计算热度
df=data[['month','tag','heat','content']]
df=df.sort_values(by=['tag','month','heat'],ascending=[True,True,False])
df.to_csv('./output/clusters_result.csv',index=None)         #聚类结果

#得到聚类后的类关键词
cnt=0
tag_valid=[]
cluster_keyword=[]
for i in range(len(set(tags))):
    if tags.count(i)>3: 
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
        print('第%d个有效聚类(tag=%d)关键词为：'%(cnt,i),cluster_key)
        cluster_keyword.append(' '.join(cluster_key))
tag_csv={'tags':tag_valid,'cluster_keywords':cluster_keyword}
df_tag=pd.DataFrame(tag_csv,columns=['tags', 'cluster_keywords'])
df_tag.to_csv('./output/cluster_keywords.csv',index=None)  #保存有效聚类的关键词

c={"month": dates,"keywords": kw}
kw = pd.DataFrame(c,columns=['month', 'keywords']) 
kw.to_csv('./output/period_keywords.csv',index=None)        #输出每个时间窗的关键词
data['splitword']=data['splitword'].apply(lambda x: ' '.join(x))
data[['time','content','splitword']].to_csv('./output/content&splitword.csv',index=None)
