#SQL
import torch
from utils import *
import numpy as np
import os
import pymysql
import random
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity

roberta = torch.hub.load(r'D:\CYW\code\dataSimiliary\roberta/fairseq', 'roberta.large',source='local')
#roberta = torch.hub.load(r'F:\Lan\project\python\LinkRecover\pytorch/fairseq', 'roberta.large.mnli',source='local')
roberta.eval()
print("模型成功")
#连接数据库
db = pymysql.connect(host="localhost", user="root", password="root", port=3306, db="sotorrent20_12_2")
def searchData(JsonPath, num, random_num):
    filed_id = 'Id'
    table_name = 'posts' #节点数据
    filed_PostTypeId = 'PostTypeId'
    filed_AnswerCount = 'AnswerCount'
    PostTypeId = 1
    answerNum = num
    filed_body = 'Body'
    q_cursor = db.cursor()
    sql_question = "SELECT {table_name}.{filed_id}, {table_name}.{filed_body} " \
              "FROM {table_name} " \
              "WHERE {table_name}.{filed_PostTypeId} = {PostTypeId} and {table_name}.{filed_AnswerCount} = {answerNum}".\
            format(table_name=table_name, filed_id=filed_id, filed_body=filed_body,
                   filed_PostTypeId=filed_PostTypeId, PostTypeId=PostTypeId, filed_AnswerCount=filed_AnswerCount, answerNum=answerNum)
    count = q_cursor.execute(sql_question)
    similityDict = {}
    jsonFileNum = 0
    totalEleNum = 0
    #获取随机数,如果不满，就全部计算
    if count > random_num:
        #随机问题ID的浮标
        random_couList = random.sample(range(0, count), random_num)
        random_cou = len(random_couList)
    else:
        random_couList = []
        random_cou = count
    #获得问题ID
    # random_couList = []
    # for index in range(10):
    #     QADict = json2Dict("./randomData/{}/Json/{}.json".format(num, index + 1))
    #     for key in QADict.keys():
    #         random_couList.append(int(key))

    for cou in range(count):
        result = q_cursor.fetchone()
        if len(random_couList) > 0:
            if cou not in random_couList:
                continue
        totalEleNum += 1
        print("遍历到第{}个问题ID{},浮标为{}".format(totalEleNum, result[0], cou))
        # if cou < finishNum:
        #     continue
        # q_feature = extractFeature(result[1])
        one_similityList = searchAnswerData(result[0], [])
        similityDict[result[0]] = one_similityList
        if totalEleNum % 1000 == 0:
            jsonFileNum += 1
            dict2Json(similityDict, JsonPath + "/{}.json".format(jsonFileNum))
            similityDict = {}
        # elif totalEleNum == len(random_couList):
        elif totalEleNum == random_cou:
            jsonFileNum += 1
            dict2Json(similityDict, JsonPath + "/{}.json".format(jsonFileNum))
            similityDict = {}
        print("第{}个问题ID处理完毕".format(totalEleNum))

    print("答案个数为{}的问题个数为{}".format(num, totalEleNum))
    q_cursor.close()

def searchDataList():
    idList = []#csv2list()
    filed_id = 'Id'
    table_name = 'posts'
    filed_body = 'Body'
    for QId in idList:
        sql_question = "SELECT {table_name}.{filed_body} " \
                  "FROM {table_name} " \
                  "WHERE {table_name}.{filed_id} = {QId}".\
                format(table_name=table_name, filed_id=filed_id, filed_body=filed_body,QId=QId)
        count = cursor.execute(sql_question)
        similityDict = {}
        eleNum = 0
        jsonFileNum = 0
        for cou in range(count):
            result = cursor.fetchone()
            q_feature = extractFeature(result[0])
            one_similityList = searchAnswerData(QId, q_feature)
            similityDict[QId] = one_similityList
            eleNum += 1
            if eleNum % 1000 == 0:
                jsonFileNum += 1
                dict2Json(similityDict, JsonPath + "/{}.json".format(jsonFileNum))
                similityDict = {}
                eleNum = 0
                # 采用随机元素
            elif totalEleNum == len(random_couList):
            #elif cou == count - 1:
                jsonFileNum += 1
                dict2Json(similityDict, JsonPath + "/{}.json".format(jsonFileNum))
                similityDict = {}
                eleNum = 0
            print("第{}个问题ID处理完毕".format(cou + 1))

def searchAnswerData(id, qid_feature):
    filed_id = "Id"
    filed_body = 'Body'
    table_name = 'posts'
    filed_ParentId = 'ParentId'
    questionId = id
    #问题、答案和相似度
    simResult = []
    a_cursor = db.cursor()
    sql_answer = "SELECT {table_name}.{filed_id}, {table_name}.{filed_body} " \
                   "FROM {table_name} " \
                   "WHERE {table_name}.{filed_ParentId} = {questionId}". \
        format(table_name=table_name, filed_id=filed_id, filed_body=filed_body,
               filed_ParentId=filed_ParentId, questionId=questionId)
    count = a_cursor.execute(sql_answer)
    #记录回答节点的特征json
    answerFeatureDict = {}
    for _ in range(count):
        result = a_cursor.fetchone()
        #计算节点特征
        aid_feature = extractFeature(result[1])
        if len(answerFeatureDict) > 0:
            #遍历已有节点
            for ans_id in answerFeatureDict:
                #计算相似度
                t_simility = calSimility(answerFeatureDict[ans_id], aid_feature)
                t_norm_simility = t_simility[0][0]
                # 保存相似度
                answerDict = {"answer_1": ans_id, "answer_2": result[0], "simility": t_norm_simility}
                simResult.append(answerDict)
        # 保存节点特征
        answerFeatureDict[result[0]] = aid_feature
        #simility = calSimility(qid_feature,aid_feature)
        #norm_simility = simility[0][0] #simility[0] / len(qid_feature)
        #answerDict = {"answer_1" : result[0], "simility" : norm_simility}
        #simResult.append(answerDict)
    a_cursor.close()
    return simResult

def extractFeature(text):
    # #初始化模型
    # roberta = torch.hub.load(r'D:\CYW\code\dataBody\roberta/fairseq', 'roberta.large',source='local')
    # #roberta = torch.hub.load(r'F:\Lan\project\python\LinkRecover\pytorch/fairseq', 'roberta.large.mnli',source='local')
    # roberta.eval()
    # 每个节点特征
    tokens = roberta.encode(text)
    if len(tokens) > 512:
        tokens = tokens[:512]
    # print(tokens)
    # Extract the last layer's features
    last_layer_features = roberta.extract_features(tokens)
    feature = last_layer_features[0][0].detach().numpy()
    return feature

def calSimility(q_feature,a_feature):
    # def sigmoid(x):
    #     return 1 / (1 + np.exp(-x))

    #adj_simility = np.dot(q_feature, a_feature)
    #adj_sim_sig = sigmoid(adj_simility)
    adj_simiality = cosine_similarity(np.array(q_feature).reshape(1, -1), np.array(a_feature).reshape(1, -1))
    return adj_simiality#, adj_sim_sig

numData = csv2List("./data_answer_simility/answerNumber.csv", ["answerNum", "totalNum", "radio", "i", "j"])
numList = []
random_num = 10000
for str_num in numData:
    numList.append(int(str_num))
for num in numList:
    if num > 2:
        JsonPath = "./data_answer_simility/{}/Json".format(num)
        if not os.path.exists(JsonPath):
            os.makedirs(JsonPath)
        searchData(JsonPath, num, random_num)

# 关闭数据库连接
db.close()
