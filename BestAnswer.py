from sklearn.metrics.pairwise import cosine_similarity
import SearchMap
import Map
import pandas as pd
import os
import re
from nltk.tokenize import sent_tokenize
from sklearn.cluster import KMeans
import numpy as np
import torch
from dataSummary import dataSummary
from utils import *


def generate_best_answer():

    def sub(s):
        s = re.sub(pattern='##########', repl='', string=s)
        s = re.sub(pattern='<a.*?>|</?[pa]>|</?strong>|</?blockquote>|</?code>|</?h['
                           '1-5]>|<img.*?>|</?ul>|</?li>|<br>|</?em>', repl='', string=s)
        return s

    def averageT():
        featureList = []
        for node in nodelist:
            featureList.append(node["feature"])

        k_means = KMeans(n_clusters=1)
        result = k_means.fit(featureList)
        cluster_center = result.cluster_centers_
        cluster_label = result.labels_
        # 根据聚类结果计算相似度
        similityList = []
        id_label = []
        for index_1 in range(len(cluster_label)):
            id_label.append([idList[index_1], cluster_label[index_1]])
            for index_2 in range(len(cluster_label)):
                # 防止不同节点重复计算和同一节点计算
                if index_1 > index_2:
                    if cluster_label[index_1] == cluster_label[index_2]:
                        t_simility = calSimility(featureList[index_1], featureList[index_2])
                        similityList.append(t_simility[0][0])
        simDataDict = {}
        t_SimMean = np.mean(similityList)

        return t_simility

    roberta = torch.hub.load(r'D:\CYW\roberta/fairseq', 'roberta.large',source='local')
    # roberta = torch.hub.load(r'F:\Lan\project\python\LinkRecover\pytorch/fairseq', 'roberta.large.mnli',source='local')
    roberta.eval()

    rd = SearchMap.Search()
    nodelist = []
    for rd_id in rd:
        df = pd.read_sql('select * from posts where id = ' + str(rd_id) + ';', con=Map.conn)
        node = {"id": df["Id"][0], "body": "", "feature": ""}
        s = df["Body"][0]

        patCode = '<pre><code>.*?</code></pre>'
        s = s.replace('\r', ' ').replace('\n', ' ').replace('\r\n', ' ')
        s = re.sub(pattern=patCode, repl='', string=s)

        s = re.sub(pattern='</?[pa]>|</?strong>|</?blockquote>|</?code>|</?h['
                           '1-5]>|</?ul>|</?li>|<br>|</?em>',
                   repl=' ', string=s)
        sens = sent_tokenize(s.replace('\n', ' '))
        node["body"] = sens.lower()
        nodelist.append(node)

    for node in nodelist:

        id = node["id"]
        body = node["body"]

        tokens = roberta.encode(body)
        if len(tokens) > 512:
            tokens = tokens[:512]

        last_layer_features = roberta.extract_features(tokens)
        feat = last_layer_features[0][0].detach().numpy()
        node["feature"] = feat

    # 读取特征向量
    featureList = []
    idList = []
    for node in nodelist:
        idList.append(node["id"])
        featureList.append(node["feature"])
    cluster_num = 2
    calMean = 0
    # 迭代条件
    # 读取作为标准的平均值
    mean = averageT()
    while mean > calMean:
        # k-means算法进行聚类
        k_means = KMeans(n_clusters=cluster_num)
        result = k_means.fit(featureList)
        cluster_center = result.cluster_centers_
        cluster_label = result.labels_
        # 根据聚类结果计算相似度
        similityList = []
        id_label = []
        for index_1 in range(len(cluster_label)):
            id_label.append([idList[index_1], cluster_label[index_1]])
            for index_2 in range(len(cluster_label)):
                # 防止不同节点重复计算和同一节点计算
                if index_1 > index_2:
                    if cluster_label[index_1] == cluster_label[index_2]:
                        t_simility = calSimility(featureList[index_1], featureList[index_2])
                        similityList.append(t_simility[0][0])
        simDataDict = {}
        t_SimMax = np.max(similityList)
        t_SimMin = np.min(similityList)
        t_SimMean = np.mean(similityList)
        t_SimVar = np.var(similityList)
        t_SimStd = np.std(similityList, ddof=1)
        simDataDict["最大值"] = t_SimMax
        simDataDict["最小值"] = t_SimMin
        simDataDict["平均值"] = t_SimMean
        simDataDict["方差"] = t_SimVar
        simDataDict["标准差"] = t_SimStd

        #聚类个数加1
        calMean = t_SimMean
        cluster_num += 1
    print("划分类的数量为{}".format(cluster_num - 1))

    cluster = {"body": "", "summaryBody":""}
    cluster_list = []
    for i in range(cluster_num - 1):
        if id_label["label"] == i:
            for node in nodelist:
                if node["id"] == id_label["id"]:
                    cluster["body"] = cluster["body"] + node["body"]
                    break
        cluster_list.append(cluster)

    cluster_list = dataSummary(cluster_list)

    best_answer = ""
    for cluster in cluster_list:
        best_answer = best_answer + cluster["summaryBody"]

    return best_answer
