import pandas as pd
#import json
import re
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity

def csv2List(csv_path, columns):
    df_data = pd.read_csv(csv_path, names=columns, header=None)
    data = []
    for df_index in range(len(df_data)):
        one_data = df_data.iloc[df_index][columns[0]]
        data.append(one_data)
    return data

#DataFrame格式转为list格式
#输入：CSV数据DataFrame， 列名
#return:数据列表
def csv2Array(csv_path, columns, dtype = ""):

    df_data = pd.read_csv(csv_path, names=columns, header=None, dtype=str, encoding="utf_8_sig")
    data = []

    for df_index in range(len(df_data)):
        onedata = []
        for col_index in range(len(columns)):
            onedata.append(df_data.iloc[df_index][columns[col_index]])
        data.append(onedata)
    return data

def list2csv(data, path, columns):
    df = pd.DataFrame(data, columns=columns)
    csv = df.to_csv(path, index=False)
    return path

def listTransCsv(data, path):
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    return path

def CsvTransArray(path, csv_encode="utf-8_sig", csv_dtype="str"):
    data = pd.read_csv(path, encoding=csv_encode, dtype=csv_dtype)
    column_list = [column for column in data]
    return data, column_list

def getTexts(text):
    text = str(text)
    text = text.replace('[', ' [ ').replace('(', ' ( ').replace('.', ' . ').replace('{', ' { ')\
        .replace('+', ' + ').replace('-', ' - ').replace('^', ' ^ ').replace('/', ' / ')\
        .replace('*', ' * ').replace('\\', ' \\ ').replace('=', ' = ').replace('！', ' ！ ')\
        .replace(']', ' ] ').replace(')', ' ) ').replace('}', ' } ').replace(':', ' : ')\
        .replace(',', ' , ').replace('_', ' _ ').replace('?', ' ? ').replace('#', ' # ') \
        .replace(';', ' ; ')
    words = text.split()
    return words

def extractword4Text(data):
    words_set = set(data)
    dict = {}
    for word_item in words_set:
        dict.update({word_item: data.count(word_item)})
    sorted_words = sorted(dict.items(), key=lambda d: d[1], reverse=True)
    extractText = ''
    wordList = []
    extractNum = 512
    if len(sorted_words) <= 512:
        extractNum = len(sorted_words)
    for index in range(extractNum):
        dict_word = sorted_words[index]
        wordList.append(dict_word[0])
        print(dict_word[0])
        extractText = extractText + dict_word[0] + ' '
    return extractText


def dealDataRedundancy(str_body):
    body = re.sub(pattern='</?[pa]>|</?strong>|</?blockquote>|</?code>|</?h[''1-5]>|<img.*?>|<a.*?>|</?ul>|</?li>|<br>|</?em>',
               repl=' ', string=str_body)
    return body

def calSimility(q_feature,a_feature):
    adj_simiality = cosine_similarity(np.array(q_feature).reshape(1, -1), np.array(a_feature).reshape(1, -1))
    return adj_simiality
