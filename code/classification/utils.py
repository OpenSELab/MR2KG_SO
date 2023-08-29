import pandas as pd
import json
import numpy as np

def csv2List(csv_path, columns):
    df_data = pd.read_csv(csv_path, names=columns, header=None,dtype=str, encoding="utf_8_sig")
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
    csv = df.to_csv(path, index=False, header=None)
    return path

def listTransCsv(data, path):
    df = pd.DataFrame(data)
    df.to_csv(path, index=False, header=None)
    return path

def CsvTransArray(path):
    data = pd.read_csv(path, names=None, header=None)
    return data

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

def json2Dict(path):
    with open(path, 'r', encoding='utf-8_sig') as load_f:
        load_dict = json.load(load_f)
    load_f.close()
    return load_dict

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def dict2Json(dict, path):
    with open(path, 'w', encoding='utf-8_sig') as write_f:
        json.dump(dict, write_f, indent=4, ensure_ascii=False, cls=NpEncoder)
    write_f.close()