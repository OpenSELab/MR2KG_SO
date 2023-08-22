from __future__ import absolute_import, division, print_function
import argparse
import os
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from preprocessor import textProcess

# epoch数
epoch = 20
# 测试batch_size和训练batch_size,如果出现内存错误请逐渐降低为4、2、1
eval_batch_size = 8
train_batch_size = 4
# 是否需要训练，是否需要测试
flag_train = True
flag_test = True
# 随机种子
seed_num = 42
# 预训练模型路径   D:\Lan\LinkRecover\CodeBERT   D:\Lan\LinkRecover\BERT     D:\Lan\LinkRecover\roberta-large
model_path = r'D:\Lan\LinkRecover\BERT'
cross = '2'

item = 'combine4'
class_num = 7
# 下面两行为先前的设置_s
# item = 'couple3'
# class_num = 5
# 类别数   注：couple3.csv有5个分类，而complete2.csv有四个分类

# item = 'complete2'
# class_num = 4


# 用于获取传参，得到参数args
def getargs():
    parser = argparse.ArgumentParser()
    # 所有参数均在最上方完成定义，下面的参数一般都不需要修改，如果修改，往往只需要在上面修改即可

    # parser.add_argument("--data_dir", default='dataset/' + cross, type=str,
    #                     help="data_dir")
    # parser.add_argument("--file", default=item, type=str,
    #                     help="The file name.")
    # parser.add_argument("--output_dir", default='saved_models/' + cross + item, type=str,
    #                     help="The output directory where the model predictions and checkpoints will be written.")
    # parser.add_argument("--result_dir", default='results/' + cross + item, type=str,
    #                     help="The result directory.")

    parser.add_argument("--data_dir", default='dataset', type=str,
                        help="data_dir")
    parser.add_argument("--file", default=item, type=str,
                        help="The file name.")
    parser.add_argument("--output_dir", default='saved_models/' + item, type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--result_dir", default='results_sep/' + item, type=str,
                        help="The result directory.")
    parser.add_argument("--model_name_or_path", default=model_path, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--tokenizer_name", default=model_path, type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--do_train", default=flag_train, type=bool,
                        help="Whether to run training.")
    parser.add_argument("--do_eval", default=True, type=bool,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", default=flag_test, type=bool,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size", default=train_batch_size, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=eval_batch_size, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed', type=int, default=seed_num,
                        help="random seed for initialization")
    parser.add_argument('--num_train_epochs', type=int, default=epoch,
                        help="num_train_epochs")
    parser.add_argument("--num_class", default=class_num, type=int,
                        help="The number of classes.")
    args = parser.parse_args()
    # 获取电脑是否有GPU，如果没有GPU就使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 如果电脑有多块GPU，则在后续进行分布式训练
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    return args


# 种子函数，用以保证结果可再现
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# 用以输入模型的单个样本类
class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 label,
                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label = label


# 用于实现将样本转换为输入特征，即上述的InputFeatures类
def convert_examples_to_features(row, tokenizer, col, args):
    text = row[col]
    # text = textProcess(row[col])
    # 使用BERT的分词器进行分词，得到BERT词汇表中的词
    tokens = tokenizer.tokenize(text)
    # 由于BERT需要在句子之间添加特殊符号[SEP]，需要在句首添加[CLS]，因而长度要保证不超过最大长度-2，否则将会截断
    if len(tokens) > args.max_seq_length - 2:
        tokens = tokens[:args.max_seq_length - 2]
    # 得到BERT可识别的token，并转换为其在词汇表中的id
    combined_token = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
    combined_ids = tokenizer.convert_tokens_to_ids(combined_token)
    # 保证长度在512，如果未达到最大长度则使用[PAD]进行补齐
    if len(combined_ids) < args.max_seq_length:
        padding_length = args.max_seq_length - len(combined_ids)
        combined_ids += [tokenizer.pad_token_id] * padding_length
    # 构建InputFeatures类作为返回值，主要包括id和label
    return InputFeatures(combined_token, combined_ids, row['label'])


# 用于实现将样本转换为输入特征，即上述的InputFeatures类
def convert_examples_to_features1(row, tokenizer, args):
    issue_text = str(row['Target_Sentence'])
    commit_text = str(row['Before_Sentence']) + ' ' + str(row['Next_Sentence'])
    issue_token = tokenizer.tokenize(issue_text)
    commit_token = tokenizer.tokenize(commit_text)
    if len(issue_token) + len(commit_token) > args.max_seq_length - 3:
        if len(issue_token) > (args.max_seq_length - 3) / 2 and len(commit_token) > (args.max_seq_length - 3) / 2:
            issue_token = issue_token[:int((args.max_seq_length - 3) / 2)]
            commit_token = commit_token[:args.max_seq_length - 3 - len(issue_token)]
        elif len(issue_token) > (args.max_seq_length - 3) / 2:
            issue_token = issue_token[:args.max_seq_length - 3 - len(commit_token)]
        elif len(commit_token) > (args.max_seq_length - 3) / 2:
            commit_token = commit_token[:args.max_seq_length - 3 - len(issue_token)]
    combined_token = [tokenizer.cls_token] + issue_token + [tokenizer.sep_token] + commit_token + [tokenizer.sep_token]
    combined_ids = tokenizer.convert_tokens_to_ids(combined_token)
    if len(combined_ids) < args.max_seq_length:
        padding_length = args.max_seq_length - len(combined_ids)
        combined_ids += [tokenizer.pad_token_id] * padding_length
    return InputFeatures(combined_token, combined_ids, row['label'])


# 用于实现将数据集全部转换为可输入模型的样本
class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        # example用于存储每个样本转化为InputFeatures后的结果，是文件中所有样本经过转化后的InputFeatures的列表
        self.examples = []
        df_link = pd.read_csv(file_path)
        for i_row, row in df_link.iterrows():
            # self.examples.append(convert_examples_to_features1(row, tokenizer, args))
            self.examples.append(convert_examples_to_features(row, tokenizer, "Multi_Sentences", args))
            # self.examples.append(convert_examples_to_features(row, tokenizer, "Target_Sentence", args))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # 得到input_id和label，分别对应为下标0和下标1
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label)



