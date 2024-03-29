from __future__ import absolute_import, division, print_function

import datetime

from sklearn.metrics import precision_score, recall_score, f1_score, \
    roc_auc_score, matthews_corrcoef, brier_score_loss, confusion_matrix
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset,\
    SequentialSampler, RandomSampler, TensorDataset
from tqdm import tqdm, trange
from model import Model
from transformers import (AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification,
                          RobertaTokenizer, AutoModel, AutoTokenizer, AutoConfig)
from utils_over import getargs, set_seed, TextDataset
import warnings
warnings.filterwarnings(action='ignore')


def train(args, train_dataset, model, tokenizer):
    dfScores = pd.DataFrame(columns=['Epoch', 'Metrics', 'Score'])

    torch.set_grad_enabled(True)
    """ Train the model """
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size, num_workers=4, pin_memory=True)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    max_steps = len(train_dataloader) * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max_steps * 0.1,
                                                num_training_steps=max_steps)

    # Train!
    print("********** Running training **********")
    print("  Num examples = {}".format(len(train_dataset)))
    print("  Num Epochs = {}".format(args.num_train_epochs))
    print("  batch size = {}".format(args.train_batch_size))
    print("  Total optimization steps = {}".format(max_steps))
    best_acc = 0.0  # 用来选在dev上最好的模型
    model.zero_grad()

    model.train()
    for idx in range(args.num_train_epochs):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        losses = []
        for step, batch in enumerate(bar):
            text_inputs = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            loss, logits = model(text_inputs, labels)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            losses.append(loss.item())
            bar.set_description("epoch {} loss {}".format(idx, round(float(np.mean(losses)), 3)))
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        results = evaluate(args, model, tokenizer)
        for key, value in results.items():
            print('-'*10 + "  {} = {}".format(key, round(value, 4)))
        for key in sorted(results.keys()):
            print('-' * 10 + "  {} = {}".format(key, str(round(results[key], 4))))
            dfScores.loc[len(dfScores)] = [idx, key, str(round(results[key], 4))]

        if results['eval_acc'] >= best_acc:
            best_acc = results['eval_acc']
            print("  " + "*" * 20)
            print("  Best acc: {}".format(round(best_acc, 4)))
            print("  " + "*" * 20)
            checkpoint_prefix = 'model.bin'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            torch.save(model, output_dir)
            print("Saving model checkpoint to {}".format(output_dir))
            dfScores.loc[len(dfScores)] = [idx, '___best___', '___best___']
        dfScores.to_csv(os.path.join(args.result_dir, 'Epoch_Metrics.csv'), index=False)


def evaluate(args, model, tokenizer):
    stime = datetime.datetime.now()
    eval_output_dir = args.output_dir

    eval_dataset = TextDataset(tokenizer, args, os.path.join(args.data_dir, args.file + '_DEV.csv'))
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4,
                                 pin_memory=True)

    # Eval!
    print("***** Running evaluation *****")
    print("  Num examples = {}".format(len(eval_dataset)))
    print("  Batch size = {}".format(args.eval_batch_size))
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    labels = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        text_inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        with torch.no_grad():
            lm_loss, logit = model(text_inputs, label)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1
    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    preds = logits.argmax(-1)

    print('Predictions', preds[:25])
    print('Labels:', labels[:25])
    etime = datetime.datetime.now()

    eval_time = (etime - stime).seconds
    eval_acc = np.mean(labels == preds)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)
    eval_precision_macro = precision_score(labels, preds, average='macro')
    eval_recall_macro = recall_score(labels, preds, average='macro')
    eval_f1_macro = f1_score(labels, preds, average='macro')
    eval_precision_micro = precision_score(labels, preds, average='micro')
    eval_recall_micro = recall_score(labels, preds, average='micro')
    eval_f1_micro = f1_score(labels, preds, average='micro')

    result = {
        "eval_loss": float(perplexity),
        "eval_time": float(eval_time),
        "eval_acc": round(float(eval_acc), 4),
        "eval_precision_macro": round(eval_precision_macro, 4),
        "eval_precision_micro": round(eval_precision_micro, 4),
        "eval_recall_macro": round(eval_recall_macro, 4),
        "eval_recall_micro": round(eval_recall_micro, 4),
        "eval_f1_macro": round(eval_f1_macro, 4),
        "eval_f1_micro": round(eval_f1_micro, 4),
    }
    return result


def test(args, model, tokenizer):
    # Note that DistributedSampler samples randomly
    stime = datetime.datetime.now()
    eval_dataset = TextDataset(tokenizer, args, os.path.join(args.data_dir, args.file + '_TEST.csv'))
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    print("********** Running Test **********")
    print("  Num examples = {}".format(len(eval_dataset)))
    print("  Batch size = {}".format(args.eval_batch_size))
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    labels = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        text_inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        with torch.no_grad():
            lm_loss, logit = model(text_inputs, label)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1

    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    preds = logits.argmax(-1)
    etime = datetime.datetime.now()

    eval_time = (etime - stime).seconds
    eval_acc = np.mean(labels == preds)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)
    eval_precision_macro = precision_score(labels, preds, average='macro')
    eval_recall_macro = recall_score(labels, preds, average='macro')
    eval_f1_macro = f1_score(labels, preds, average='macro')
    eval_precision_micro = precision_score(labels, preds, average='micro')
    eval_recall_micro = recall_score(labels, preds, average='micro')
    eval_f1_micro = f1_score(labels, preds, average='micro')

    result = {
        "eval_loss": float(perplexity),
        "eval_time": float(eval_time),
        "eval_acc": round(float(eval_acc), 4),
        "eval_precision_macro": round(eval_precision_macro, 4),
        "eval_precision_micro": round(eval_precision_micro, 4),
        "eval_recall_macro": round(eval_recall_macro, 4),
        "eval_recall_micro": round(eval_recall_micro, 4),
        "eval_f1_macro": round(eval_f1_macro, 4),
        "eval_f1_micro": round(eval_f1_micro, 4),
    }
    print(preds[:25], labels[:25])
    print("********** Test results **********")
    dfScores = pd.DataFrame(columns=['Metrics', 'Score'])
    for key in sorted(result.keys()):
        print('-'*10 + "  {} = {}".format(key, str(round(result[key], 4))))
        dfScores.loc[len(dfScores)] = [key, str(round(result[key], 4))]
    dfScores.to_csv(os.path.join(args.result_dir, 'Metrics.csv'), index=False)
    assert len(logits) == len(preds) and len(logits) == len(labels), 'error'

    cols = [str(i) + '_pred' for i in range(args.num_class)]
    df = pd.DataFrame(logits, columns=cols)
    df['preds'] = preds
    df['labels'] = labels
    print("+++"*20)
    # df = pd.DataFrame(np.transpose([preds, labels]),
    #                   columns=['preds', 'labels'])
    df.to_csv(os.path.join(args.result_dir, 'predictions.csv'), index=False)


def main():
    args = getargs()
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    print("device: {}, n_gpu: {}".format(args.device, args.n_gpu) )
    # Set seed
    set_seed(args.seed)

    # 配置Roberta
    # config = RobertaConfig.from_pretrained(args.model_name_or_path)
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    config.num_labels = args.num_class
    # tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    # model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
    # textEncoder = AutoModel.from_pretrained(args.model_name_or_path, config=config)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    textEncoder = AutoModel.from_pretrained(args.model_name_or_path, config=config)

    model = Model(textEncoder, config, tokenizer, args)
    # multi-gpu training (should be after apex fp16 initialization)
    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    print("Training/evaluation parameters {}".format(args) )

    # Training
    if args.do_train:
        # sample - dataset
        train_dataset = TextDataset(tokenizer, args, os.path.join(args.data_dir, args.file + '_TRAIN.csv'))
        train(args, train_dataset, model, tokenizer)

    if args.do_test:
        checkpoint_prefix = 'model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model = torch.load(output_dir)
        model.to(args.device)
        test(args, model, tokenizer)


if __name__ == "__main__":
    print("======BEGIN======"*20)
    main()
