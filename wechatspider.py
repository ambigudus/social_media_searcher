import wechatsogou
from wechatsogou import WechatSogouAPI, WechatSogouConst
import time
ws_api = wechatsogou.WechatSogouAPI(proxies={
    "http": "127.0.0.1:9999",
    "https": "127.0.0.1:9999",
})

keywords=['娱乐', '八卦', '新闻', '明星']

def we_spider():
    for word in keywords:
        for i in range(1,10):
            time.sleep(1)
            res=ws_api.search_article(word,page=i,timesn=WechatSogouConst.search_article_time.day,article_type=WechatSogouConst.search_article_type.all)
            for j in range(0,len(res)):
                print(res[j]['article']['title'])

    time.sleep(60*60*24)##one day

we_spider()