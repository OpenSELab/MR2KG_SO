import os

import pandas as pd
import re
from nltk.tokenize import sent_tokenize

Path_Resource = r'../dataset_sun'
dic = {'couple3.csv': 'ID_reference', 'complete2.csv': 'external'}


def sub(s):
    s = re.sub(pattern='##########', repl='', string=s)
    s = re.sub(pattern='<a.*?>|</?[pa]>|</?strong>|</?blockquote>|</?code>|</?h['
                       '1-5]>|<img.*?>|</?ul>|</?li>|<br>|</?em>', repl='', string=s)
    return s


def main(it):
    ls = []
    # df = pd.read_csv(os.path.join(Path_Resource, it), encoding='GBK')
    df = pd.read_csv(os.path.join(Path_Resource, it))
    print(df.columns)
    dfnew = pd.DataFrame(
        columns=df.columns.tolist() + ['Multi_Sentences', 'Before_Sentence', 'Target_Sentence', 'Next_Sentence'])
    for i, row in df.iterrows():
        s = str(row['Body'])
        if it.startswith('couple'):
            patID = f'<a href="[^ ]*?{str(int(row[dic[it]]))}.*?>'
        elif it.startswith('complete'):
            patID = f'{str(row[dic[it]])}'.replace('.', '\.').replace('?', '\?').replace('#', '\#').replace('(',
                                                                                                            '\(').replace(
                ')', '\)').replace('+', '\+').replace('-', '\-')
            patID = re.sub('https?', repl='https?', string=patID)

        else:
            patID = ' '

        patCode = '<pre><code>.*?</code></pre>'
        s = s.replace('\r', ' ').replace('\n', ' ').replace('\r\n', ' ')
        s = re.sub(pattern=patCode, repl='', string=s)
        s = s.lower()

        if not re.search(patID, s):
            # print(row['ID'], row['class'])
            # print(s)
            # print(patID)
            # print('-'*90)

            # ls.append(str(row[dic[it]]))
            ls.append(row['ID'])
        else:
            s = re.sub(pattern=patID, repl='########## ', string=s)
            s = re.sub(pattern='</?[pa]>|</?strong>|</?blockquote>|</?code>|</?h['
                               '1-5]>|</?ul>|</?li>|<br>|</?em>',
                       repl=' ', string=s)
            sens = sent_tokenize(s.replace('\n', ' '))
            select_sens = []
            for i, sen in enumerate(sens):
                sens[i] = re.sub(pattern='</?.*?>', repl=' ', string=sen)
                # print(sens)
                if re.search('##########', sen):
                    if i != 0:
                        select_sens.append(sub(sens[i - 1]))
                    else:
                        select_sens.append(' ')
                    select_sens.append(sub(sens[i]))
                    if i != len(sens) - 1:
                        select_sens.append(sub(sens[i + 1]))
                    else:
                        select_sens.append(' ')
                    break
                else:
                    continue
            assert len(select_sens) == 3, str(row['ID']) + s
            dfnew.loc[len(dfnew)] = row.values.tolist() + [' '.join(select_sens)] + select_sens
            dfnew.to_csv(os.path.join(Path_Resource, it[: it.find(".csv")] + '_TEST.csv'), index=False)

    print(len(ls), ls)
    print(len(df), len(dfnew))


if __name__ == "__main__":
    files = ['couple3.csv', 'complete2.csv']
    # for f in reversed(files):
    for f in files:
        print('+' * 50 + f)
        main(f)
        # exit(525)
