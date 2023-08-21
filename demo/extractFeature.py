import torch
from utils import *
import numpy as np
import os

#初始化模型
roberta = torch.hub.load(r'D:\CYW\roberta/fairseq', 'roberta.large',source='local')
#roberta = torch.hub.load(r'F:\Lan\project\python\LinkRecover\pytorch/fairseq', 'roberta.large.mnli',source='local')
roberta.eval()

#迭代访问信息
filepath = './node_73_0728_QA'
idList = csv2List(filepath + '/RQ2node.csv', ['id'])
#idList = [34872979]
#结束节点个数
finishedNum = 69
#访问每一个节点id
for id_index in range(len(idList)):
    # if id_index < finishedNum:
    #     continue
    id = idList[id_index]
    #访问id的每个节点
    nodeJsonPath = filepath + '/id_{}/QACombine/QAJson/QABodyJson.json'.format(id)
    nodeDict = json2Dict(nodeJsonPath)
    nodeFinishNum = 0
    for node_key in nodeDict.keys():
        print(node_key)
        #节点summaryBody
        t_summaryBody = nodeDict[node_key]["body"]
        # 每个节点特征
        tokens = roberta.encode(t_summaryBody)
        if len(tokens) > 512:
            tokens = tokens[:512]
        # print(tokens)
        # Extract the last layer's features
        last_layer_features = roberta.extract_features(tokens)
        feat = last_layer_features[0][0].detach().numpy()
        nodeDict[node_key]["feature"] = feat
        nodeFinishNum += 1
        print("id_{}完成{}/{}".format(id, nodeFinishNum, len(nodeDict)))
    dict2Json(nodeDict, nodeJsonPath.format(id))
    finishedNum += 1
    print("完成第{}个节点id_{}".format(finishedNum, id))
