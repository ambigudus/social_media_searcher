# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 13:33:04 2019

@author: houwenxin
"""

import re
import pandas as pd
import jieba
"""
data = pd.read_excel("data_out.xls")

sources = data["来源"].values.tolist()
times = [int(re.findall(r"([0-9]{4})", source)[0]) for source in sources]
data.rename(columns={"来源":"时间"}, inplace=True)
data["时间"] = times
data = data.sort_values(by="时间",ascending=True)
times = data["时间"].values.tolist()

keywords = data["关键词"].values.tolist()

keywords_list = [keyword.split("：")[1].strip().split(" ") for keyword in keywords]

"""
def stopwordslist(filepath): 
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()] 
    return stopwords 
def seg_sentence(sentence): 
    sentence_seged = jieba.cut(sentence.strip()) 
    stopwords = stopwordslist('Mystopwords.txt') # 这里加载停用词的路径  
    outstr = [] 
    for word in sentence_seged: 
        if word not in stopwords: 
            if word != '\t': 
                outstr.append(word)
    return outstr 

times = []
keywords_list = []
with open("test.csv", "r", encoding="utf-8") as file:
    for line in file.readlines():
        elems = line.split("\t")
        times.append(int("".join(elems[0].split("_")[:3])))
        keywords_list.append(seg_sentence(elems[1].strip()))
to_sort = list(zip(times, keywords_list))

to_sort = sorted(to_sort, key=lambda x:x[0], reverse=True)
times[:], keywords_list[:] = zip(*to_sort)
print(times)

time_word_dict = {}

for i in range(len(times)):
    if times[i] in time_word_dict:
        time_word_dict[times[i]].extend(keywords_list[i])
    else:
        print(times[i])
        time_word_dict[times[i]] = keywords_list[i]
#print(time_word_dict)

print(len(time_word_dict))

for key,words in time_word_dict.items():
    word_count = {}
    for i in range(len(words)):
        word = words[i]
        if word in word_count:
            word_count[word]["F"] += 1
        else:
            word_count[word] = {"F":1, "R":0} # F：词频 Fi, R：增长率 Fi/(Fi-1 + 1)

    time_word_dict[key] = word_count

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
    F_output.append(sorted(word_dict.items(), key=lambda x:x[1]["F"], reverse=True)[:5])
    R_output.append(sorted(word_dict.items(), key=lambda x:x[1]["R"], reverse=True)[:2])
#print(len(F_output))
for i in range(len(F_output)):
    output.append([word for word in F_output[i] if word in R_output[i]])

times = list(set(times))
times.sort(reverse=True)
for i in range(len(output)):
    print("年份:", times[i])
    print("事件:", end="")
    for j in range(len(output[i])):
        print(output[i][j][0], end=" ")
    print("\n")
print(len(output))

