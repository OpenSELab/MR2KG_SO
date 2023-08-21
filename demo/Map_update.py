import pymysql
import pandas as pd
import json
import time
import csv
import cut
import os
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
import os
import re
from nltk.tokenize import sent_tokenize
from sklearn.cluster import KMeans
import numpy as np
import torch
from dataSummary import dataSummary
from utils import *


start = time.perf_counter()


def num(strs):
    for s in strs:
        if s.isdigit():
            return True
    return False


class properties:
    节点序号 = 0
    类型 = ""
    out_degree = 0
    in_degree = 0
    Type = ""
    body = ""

    def todict(self):
        return {"节点序号": self.节点序号, "类型": self.类型, "out_degree": self.out_degree, "in_degree": self.in_degree,
                "Type": self.Type, "body": self.body}


class Post:
    id = ""
    properties = properties()
    isCenter = 0
    labels = ""

    def todict(self):
        return {"id": self.id, "properties": self.properties.todict(), "isCenter": self.isCenter, "labels": self.labels}


class Edge:
    BeginId = ""
    EndId = ""
    Type = ""
    id = 0
    Type2 = ""

    def printf(self):
        print(str(self.BeginId) + " " + str(self.EndId) + " " + str(self.id))


class Link:
    url = ""


class properties2:
    源节点 = ""
    目标节点 = ""

    def todict(self):
        return {"源节点": self.源节点, "目标节点": self.目标节点}


class Edge2:
    BeginId = ""
    url = ""
    Type = ""
    id = 0
    Type2 = ""

    def printf(self):
        print(str(self.BeginId) + " " + str(self.url) + " " + str(self.id))


class Edge3:
    id = -1
    source = -1
    type = ""
    properties2 = properties2()
    target = -1
    Type2 = ""

    def printf(self):
        print(str(self.id) + " " + str(self.source) + " " + str(self.target))

    def todict(self):
        return {"id": self.id, "source": self.source, "type": self.type, "properties": self.properties2.todict(),
                "target": self.target, "Type2": self.Type2}


def Search(id, edgelistdic, conn):
    rd = []
    sn = [str(id)]

    def rd_append(k):
        if edge["properties"]["目标节点"] not in rd:
            rd.append(edge["properties"]["目标节点"])

    def sn_append(k):
        if edge["properties"]["目标节点"] not in sn:
            sn.append(edge["properties"]["目标节点"])

    while len(sn) > 0:

        sntop = sn[0]
        for edge in edgelistdic:
            if edge["properties"]["源节点"] == sntop:
                if edge["type"] == "Q-A":
                    rd_append(edge)
                    sn_append(edge)
                if edge["type"] == "duplicate":
                    df = pd.read_sql('select * from posts where id = ' + str(edge["properties"]["目标节点"]) + ';',
                                     con=conn)
                    for i in range(len(df)):
                        if df["PostTypeId"][i] == 1:
                            sn_append(edge)
                        if df["PostTypeId"][i] == 2:
                            rd_append(edge)
                            sn_append(edge)
                if edge["type"] == "4":
                    rd_append(edge)
                    sn_append(edge)
        del(sn[0])

    return rd


def generate_best_answer(rd, postlistdic):

    roberta = torch.hub.load(r'D:\CYW\roberta/fairseq', 'roberta.large', source='local')
    # roberta = torch.hub.load(r'F:\Lan\project\python\LinkRecover\pytorch/fairseq', 'roberta.large.mnli',source='local')
    roberta.eval()

    nodelist = []
    for rd_id in rd:
        node = {"id": "", "body": "", "feature": []}
        for post in postlistdic:
            if post["id"] == rd_id:
                node["id"] = post["id"]
                node["body"] = post["properties"]["body"]
        s = node["body"]

        patCode = '<pre><code>.*?</code></pre>'
        s = s.replace('\r', ' ').replace('\n', ' ').replace('\r\n', ' ')
        s = re.sub(pattern=patCode, repl='', string=s)

        s = re.sub(pattern='</?[pa]>|</?strong>|</?blockquote>|</?code>|</?h['
                           '1-5]>|</?ul>|</?li>|<br>|</?em>',
                   repl=' ', string=s)
        s = s.lower()
        sens = sent_tokenize(s.replace('\n', ' '))
        node["body"] = sens

        nodelist.append(node)

    def averageT(nodelist):
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
        t_SimMean = np.mean(similityList)

        return t_SimMean

    for node in nodelist:

        id = node["id"]
        body = node["body"]
        body = ''.join(body)

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
    mean = averageT(nodelist)
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

        if len(similityList) > 0:
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
            calMean = t_SimMin
            cluster_num += 1
        else:
            break
    print("划分类的数量为{}".format(cluster_num - 1))

    cluster = {"body": "", "summaryBody": ""}
    cluster_list = []
    for i in range(cluster_num - 1):
        cluster = {"body": "", "summaryBody": ""}
        for j in range(len(id_label)):
            if id_label[j][1] == i:
                for node in nodelist:
                    if node["id"] == id_label[j][0]:
                        cluster["body"] = cluster["body"] + ''.join(node["body"])
                        break
        cluster_list.append(cluster)

    cluster_list = dataSummary(cluster_list)

    best_answer = "Answer:<br>"
    for i in range(len(cluster_list)):
        best_answer = best_answer + str(i + 1) + "." + cluster_list[i]["summaryBody"] + "<br>"

    return best_answer

def map(inputid):
    conn = pymysql.connect(
        user='root',
        password='root',
        host='localhost',
        database='sotorrent',
        port=3306,
        charset='utf8mb4',
    )

    todolist = []
    postidlist = []
    postlist = []
    linklist = []
    edgelist = []
    edge2list = []
    edge3list = []
    senlist = []
    senqalist = []
    answers_associated_with_the_specific_question = []

    todolist.append(str(inputid))
    df0 = pd.read_sql('select * from posts where parentid = ' + str(inputid) + ';', con=conn)
    post0 = Post()
    nodenum = 0
    linknum = 0

    for j in range(len(df0)):
        post0.id = str(df0["Id"][j])
        answers_associated_with_the_specific_question.append(post0.id)
        if post0.id not in todolist and post0.id not in postidlist:
            todolist.append(post0.id)

        edge = Edge()
        edge.BeginId = inputid
        edge.EndId = post0.id
        edge.Type = "Q-A"
        edge.Type2 = "Question-answer"
        if edge not in edgelist:
            edge.id = linknum
            linknum = linknum + 1
            edgelist.append(edge)

    while len(todolist) > 0:

        todo = todolist[0]
        df = pd.read_sql('select * from posts where id = ' + str(todo) + ';', con=conn)

        if len(df) == 0:
            print(todo)
            todolist.remove(todo)
            for edge in edgelist:
                if edge.EndId == todo:
                    edgelist.remove(edge)

            continue

        body = df["Body"].tolist()
        df4 = pd.read_sql('select * from postlinks where postid = ' + str(df["Id"][0]) + ' and LinkTypeId = 3;', con=conn)
        for i in range(len(df4)):
            if str(df4["RelatedPostId"][i]) not in todolist and str(df4["RelatedPostId"][i]) not in postidlist:
                todolist.append(str(df4["RelatedPostId"][i]))

                edge = Edge()
                edge.BeginId = str(df["Id"][0])
                edge.EndId = str(df4["RelatedPostId"][i])
                edge.Type = "duplicate"
                if edge not in edgelist:
                    edge.id = linknum
                    linknum = linknum + 1
                    edgelist.append(edge)

        post = Post()
        post.id = str(df["Id"][0])

        pro = properties()
        pro.节点序号 = nodenum
        nodenum = nodenum + 1
        if df["PostTypeId"][0] == 1:
            pro.类型 = "Question"
            pro.Type = "Coupling"
        elif df["PostTypeId"][0] == 2:
            if post.id in answers_associated_with_the_specific_question:
                pro.Type = "hierarchy"
            else:
                pro.Type = "Coupling"
            pro.类型 = "answer"

        pro.body = df["Body"][0]
        post.properties = pro
        post.labels = str(df["Id"][0])

        if post not in postlist:
            postidlist.append(str(post.id))
            postlist.append(post)

        while body[0].find("<a href=\"") != -1 or body[0].find("<img src=\"") != -1:

            if body[0].find("<a href=\"") != -1:
                idx1 = body[0].find("<a href=\"")
                idx2 = body[0][idx1 + 9:].find("\"")
                url = body[0][idx1 + 9:idx1 + 9 + idx2]
            elif body[0].find("<img src=\"") != -1:
                idx1 = body[0].find("<img src=\"")
                idx2 = body[0][idx1 + 10:].find("\"")
                url = body[0][idx1 + 10:idx1 + 10 + idx2]

            url = url.lower()

            if url.find("://stackoverflow.com/") == -1:
                link = Link()
                link.url = url
                if link not in linklist:
                    linklist.append(link)

                edge2 = Edge2()
                edge2.BeginId = df["Id"][0]

                if url[url.find("//") + 2:].find("/") != -1:
                    urldomain = url[url.find("//") + 2:url.find("//") + 2 + url[url.find("//") + 2:].find("/")]
                else:
                    urldomain = url[url.find("//") + 2:]

                edge2.url = url
                edge2.Type = ""
                edge2.Type2 = "Completeness"

                if edge2 not in edge2list:
                    sen = {'ID': edge2.BeginId, 'Body': body[0], 'external': edge2.url}
                    senlist.append(sen)

                    edge2.id = linknum
                    linknum = linknum + 1
                    edge2list.append(edge2)

                    post = Post()
                    post.id = url

                    pro = properties()
                    pro.节点序号 = nodenum
                    pro.Type = "Completeness"
                    nodenum = nodenum + 1
                    pro.类型 = "external-link"
                    pro.out_degree = 0
                    pro.body = df["Body"][0]

                    post.properties = pro
                    post.labels = urldomain

                    if post not in postlist:
                        postidlist.append(str(post.id))
                        postlist.append(post)

            elif url.find("stackoverflow.com") != -1:
                if not num(url):
                    body[0] = body[0][idx1 + 9 + idx2:]
                    continue
                if url.find("stackoverflow.com/q") != -1 and (
                        url[url.find("stackoverflow.com/q"):].find("#") == -1 or url[
                                                                                 url.find("stackoverflow.com/q"):].find(
                        "?sort") != -1 or url[url.find("stackoverflow.com/q"):].find("#") == len(
                        url[url.find("stackoverflow.com/q"):]) - 1):
                    post = Post()
                    if url.find("stackoverflow.com/question") != -1:
                        if url[url.find("stackoverflow.com/questions/") + 28:].find("/") != -1:
                            post.id = (url[url.find("stackoverflow.com/questions/") + 28:url.find(
                                "stackoverflow.com/questions/") + 28 + url[url.find(
                                "stackoverflow.com/questions/") + 28:].find("/")])
                        elif url[url.find("stackoverflow.com/questions/") + 28:].find("?sort") != -1:
                            post.id = (url[url.find("stackoverflow.com/questions/") + 28:url.find(
                                "stackoverflow.com/questions/") + 28 + url[url.find(
                                "stackoverflow.com/questions/") + 28:].find("?")])
                        else:
                            post.id = (url[url.find("stackoverflow.com/questions/") + 28:])

                    elif url.find("stackoverflow.com/q") != -1:
                        if url[url.find("stackoverflow.com/q/") + 20:].find("/") != -1:
                            post.id = (url[url.find("stackoverflow.com/q/") + 20:url.find(
                                "stackoverflow.com/q/") + 20 + url[url.find("stackoverflow.com/q/") + 20:].find("/")])
                        elif url[url.find("stackoverflow.com/q/") + 20:].find("?sort") != -1:
                            post.id = (url[url.find("stackoverflow.com/q/") + 20:url.find(
                                "stackoverflow.com/q/") + 20 + url[url.find("stackoverflow.com/q/") + 20:].find("?")])
                        else:
                            post.id = (url[url.find("stackoverflow.com/q/") + 20:])

                    if post.id not in todolist and post.id not in postidlist:
                        todolist.append(post.id)

                    edge = Edge()
                    edge.BeginId = df["Id"][0]
                    edge.EndId = post.id
                    edge.Type = ""
                    edge.Type2 = "coupling"
                    if edge not in edgelist:
                        edge.id = linknum
                        linknum = linknum + 1
                        edgelist.append(edge)
                        senqa = {'ID': edge.BeginId, 'Body': body[0], 'ID_reference': edge.EndId}
                        senqalist.append(senqa)

                    df2 = pd.read_sql('select * from posts where parentid = ' + post.id + ';', con=conn)
                    post2 = Post()
                    for j in range(len(df2)):
                        post2.id = str(df2["Id"][j])
                        if post2.id not in todolist and post2.id not in postidlist:
                            todolist.append(post2.id)

                        edge = Edge()
                        edge.BeginId = post.id
                        edge.EndId = post2.id
                        edge.Type = "Q-A"
                        edge.Type2 = "Question-answer2"
                        if edge not in edgelist:
                            edge.id = linknum
                            linknum = linknum + 1
                            edgelist.append(edge)

                elif url.find("stackoverflow.com/a/") != -1 or url.find("stackoverflow.com/q") != -1 and url[url.find(
                        "stackoverflow.com/q"):].find("#") != -1 and url[url.find("stackoverflow.com/q"):].find(
                        "#comment") == -1:
                    if url.find("stackoverflow.com/a/") != -1:
                        post = Post()
                        if url[url.find("stackoverflow.com/a/") + 20:].find("/") != -1:
                            post.id = (url[url.find("stackoverflow.com/a/") + 20:url.find("stackoverflow.com/a/") + 20 + url[url.find("stackoverflow.com/a/") + 20:].find("/")])
                        else:
                            post.id = (url[url.find("stackoverflow.com/a/") + 20:])
                        if post.id not in todolist and post.id not in postidlist:
                            todolist.append(post.id)

                        edge = Edge()
                        edge.BeginId = df["Id"][0]
                        edge.EndId = post.id
                        edge.Type = ""
                        edge.Type2 = "coupling"
                        if edge not in edgelist:
                            edge.id = linknum
                            linknum = linknum + 1
                            edgelist.append(edge)
                            senqa = {'ID': edge.BeginId, 'Body': body[0], 'ID_reference': edge.EndId}
                            senqalist.append(senqa)

                    else:
                        post2 = Post()
                        if url[url.find("#") + 1:].find("-") != -1:
                            post2.id = (url[url[url.find("#") + 1:].find("-") + 2 + url.find("#"):])
                        else:
                            post2.id = (url[url.find("#") + 1:])
                        post2.PostTypeId = 2
                        if post2.id not in todolist and post2.id not in postidlist:
                            todolist.append(post2.id)

                        edge = Edge()
                        edge.BeginId = df["Id"][0]
                        edge.EndId = post2.id
                        edge.Type = ""
                        if edge not in edgelist:
                            edge.id = linknum
                            linknum = linknum + 1
                            edgelist.append(edge)
                            senqa = {'ID': edge.BeginId, 'Body': body[0], 'ID_reference': edge.EndId}
                            senqalist.append(senqa)

            body[0] = body[0][idx1 + 9 + idx2:]

        todolist.remove(todo)

    header = ['ID', 'Body', 'external', 'label']
    with open('..//dataset_sun//complete2.csv', 'w', encoding='UTF8', newline='') as g:
        writer = csv.writer(g)
        writer.writerow(header)
        for i in range(len(senlist)):
            writer.writerow([senlist[i]["ID"], senlist[i]["Body"], senlist[i]["external"], 0])
    g.close()

    header = ['ID', 'Body', 'ID_reference', 'label']
    with open('..//dataset_sun//couple3.csv', 'w', encoding='UTF8', newline='') as g:
        writer = csv.writer(g)
        writer.writerow(header)
        for i in range(len(senqalist)):
            writer.writerow([senqalist[i]["ID"], senqalist[i]["Body"], senqalist[i]["ID_reference"], 0])
    g.close()

    conn.close()

    files = ['couple3.csv', 'complete2.csv']
    # for f in reversed(files):
    for f in files:
        print('+' * 50 + f)
        cut.main(f)

    os.system(r'../run_sun.sh')

    df_precou = pd.read_csv("..//results_sep_sun//couple3//predictions.csv")
    df_precom = pd.read_csv("..//results_sep_sun//complete2//predictions.csv")
    df_cou = pd.read_csv("..//dataset_sun//couple3_TEST.csv")
    df_com = pd.read_csv("..//dataset_sun//complete2_TEST.csv")
    df_cou["pred"] = df_precou["preds"]
    df_com["pred"] = df_precom["preds"]
    df_cou.to_csv("..//dataset_sun//couple3_TEST.csv", mode='w')
    df_com.to_csv("..//dataset_sun//complete2_TEST.csv", mode='w')

    df_cou = pd.read_csv("..//dataset_sun//couple3_TEST.csv")
    for i in range(len(edgelist)):
        for j in range(len(df_cou)):
            if str(edgelist[i].BeginId) == str(df_cou["ID"][j]) and str(edgelist[i].EndId) == str(df_cou["ID_reference"][j]):
                edgelist[i].Type = str(df_cou["pred"][j])

    df_com = pd.read_csv("..//dataset_sun//complete2_TEST.csv")
    for i in range(len(edge2list)):
        for j in range(len(df_com)):
            if str(edge2list[i].BeginId) == str(df_com["ID"][j]) and str(edge2list[i].url) == str(df_com["external"][j]):
                edge2list[i].Type = str(df_com["pred"][j])

    postlistdic = []
    edgelistdic = []
    maxd = -1
    maxi = -1
    for i in range(len(postlist)):
        if postlist[i].properties.类型 == "Question" or postlist[i].properties.类型 == "answer":
            de = 0
            for j in range(len(edgelist)):
                if str(edgelist[j].BeginId) == str(postlist[i].id):
                    de = de + 1
            for j in range(len(edge2list)):
                if str(edge2list[j].BeginId) == str(postlist[i].id):
                    de = de + 1
            postlist[i].properties.out_degree = de

            de = 0
            for j in range(len(edgelist)):
                if str(edgelist[j].EndId) == str(postlist[i].id):
                    de = de + 1
            postlist[i].properties.in_degree = de

        elif postlist[i].properties.类型 == "external-link":
            de = 0
            for j in range(len(edge2list)):
                if str(edge2list[j].url) == str(postlist[i].id):
                    de = de + 1
            postlist[i].properties.in_degree = de

        if postlist[i].properties.in_degree + postlist[i].properties.out_degree > maxd:
            maxd = postlist[i].properties.in_degree + postlist[i].properties.out_degree
            maxi = i

        if postlist[i].properties.in_degree + postlist[i].properties.out_degree == 0:
            postlist.remove(postlist[i])

    postlist[0].isCenter = 1

    for i in range(len(postlist)):
        postlistdic.append(postlist[i].todict())

    for i in range(len(edgelist)):
        edge4 = Edge3()
        edge4.id = edgelist[i].id

        for j in range(len(postlist)):
            if str(edgelist[i].BeginId) == postlist[j].id:
                edge4.source = postlist[j].properties.节点序号
                break

        for j in range(len(postlist)):
            if str(edgelist[i].EndId) == postlist[j].id:
                edge4.target = postlist[j].properties.节点序号
                break

        edge4.type = edgelist[i].Type
        edge4.Type2 = edgelist[i].Type2
        edge4.properties2.源节点 = str(edgelist[i].BeginId)
        edge4.properties2.目标节点 = str(edgelist[i].EndId)

        if edge4 not in edge3list:
            edge3list.append(edge4)

        edgelistdic.append(edge3list[i].todict())

    for i in range(len(edge2list)):
        edge3 = Edge3()
        edge3.id = edge2list[i].id

        for j in range(len(postlist)):
            if str(edge2list[i].BeginId) == postlist[j].id:
                edge3.source = postlist[j].properties.节点序号
                break

        for j in range(len(postlist)):
            if str(edge2list[i].url) == postlist[j].id:
                edge3.target = postlist[j].properties.节点序号
                break

        edge3.type = edge2list[i].Type
        edge3.Type2 = edge2list[i].Type2
        edge3.properties2.源节点 = str(edge2list[i].BeginId)
        edge3.properties2.目标节点 = str(edge2list[i].url)

        if edge3 not in edge3list:
            edge3list.append(edge3)

        edgelistdic.append(edge3list[len(edgelist) + i].todict())

    for post in postlistdic:
        post["properties"]["body"] = post["properties"]["body"].replace('\n', '')


    data = {"nodes": postlistdic, "links": edgelistdic}
    jsOnStr = json.dumps(data, indent=1, ensure_ascii=False)

    f = open("result.txt", "w", encoding='utf-8')
    f.write(jsOnStr)
    f.close()

    return edgelistdic, conn, postlistdic


id = input("请输入需要搜索的节点Post的Id：")
edgelistdic, conn, postlistdic = map(id)
print("Map Complete.")

rd = Search(id, edgelistdic, conn)
print("Search Complete.")
best_answer = generate_best_answer(rd, postlistdic)
print(best_answer)

f = open("best_answer.txt", "w", encoding='utf-8')
f.write(best_answer)
f.close()

end = time.perf_counter()
print('Running time: %s Seconds' % (end - start))

