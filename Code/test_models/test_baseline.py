import os
import re
import sys
import time
import json
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

sys.path.append('..')
from models import baseline_models


def get_label(line, comment_max_seq_length, tag2id, file_num):
    """
    获得每句话的标签
    :param line: str
            comment
    :param comment_max_seq_length: int
            文本最大长度
    :param tag2id: dict
    :return label: list
    """
    text = line.rstrip('\n').split('\t')[1]
    text = re.sub('\[CLS\] ', '', text)
    text = re.sub(' \[SEP\]', '', text)
    label = [0] * comment_max_seq_length
    tokens = text.split(' ')

    for i in range(len(tokens)):
        if re.search(r'\\[BI]', tokens[i]):
            try:
                label[i] = tag2id[tokens[i][-1]]
            except KeyError:
                print(file_num, text)

    return label


def read_data(file_path, comment_max_seq_length, tag2id):
    """
    读取文本数据与标签数据
    :param file_path: str
            文件路径
    :param comment_max_seq_length: int
            文本最大长度
    :param tag2id: dict
    :return inputs: list
            [weibo1[SEP]comment11, weibo1[SEP]comment12, weibo2[SEP]comment21, ...]
    :return labels: list
            只有目标词语的位置为 1, 其余位置为0
    """
    content_path = os.path.join(file_path, 'content')
    label_path = os.path.join(file_path, 'label')
    assert len(os.listdir(content_path)) == len(os.listdir(label_path))

    files = os.listdir(content_path)
    file_nums = [file.split('.')[0].split('_')[1] for file in files]

    inputs = []
    labels = []
    for file_num in file_nums:
        df = pd.read_csv(os.path.join(content_path, 'content_{}.pkl.csv'.format(file_num)),
                         names=['label', 'content'])
        contents = df['content'].tolist()

        with open(os.path.join(label_path, 'content_{}.txt'.format(file_num)),
                  'r', encoding='utf-8-sig') as f:
            lines = f.readlines()

        assert len(contents) == len(lines)

        for j in range(len(contents)):
            # 排除微博正文
            if j == 0:
                continue
            else:
                inputs.append(contents[j])
                labels.append(get_label(lines[j], comment_max_seq_length, tag2id, file_num))

    return inputs, labels


def trunc_and_padding(texts, config, vocab):
    """
    截断与填充, 获得 mask, 并将 token 转为 id
    :param texts:
    :param config:
    :param vocab:
    :return:
    """
    indices, masks = [], []
    for text in texts:
        words = list(text)
        ids = [vocab['PAD']] * config.max_seq_length
        mask = [0] * config.max_seq_length
        if len(words) > config.max_seq_length:
            words = words[:config.max_seq_length]
        i = 0
        for word in words:
            if vocab.__contains__(word):
                ids[i] = vocab[word]
            else:
                ids[i] = vocab['UNK']
            mask[i] = 1
            i += 1
        indices.append(ids)
        masks.append(mask)

    return torch.LongTensor(indices), torch.LongTensor(masks)


def convert_id2tag(labels, id2tag):
    """
    将 id 转换为 tag
    :param labels: list
    :param id2tag: dict
    :return:
    """
    tag_labels = []
    for i in range(len(labels)):
        tag_labels.append([])
        for id_ in labels[i]:
            tag_labels[i].append(id2tag[id_])

    return tag_labels


def evaluate(model, data_iter, device, mode, id2tag):
    """
    Evaluating model on dev and test, and outputting loss, accuracy and macro-F1
    :param model: Object
    :param data_iter: DataLoader
            dev or test
    :param device: Object
    :return: float
            total loss
    :return acc: float
            total accuracy
    :return macro_F1: float
            total macro-F1
    """
    model.eval()
    loss_total = 0
    predict_all = []
    labels_all = []

    with torch.no_grad():
        for texts, labels, masks in data_iter:
            texts = texts.to(device)
            labels = labels.to(device)
            masks = masks.to(device)



            loss, logits = model(texts, masks, labels, mode)
            # loss = F.cross_entropy(outputs, labels)
            loss_total += loss

            pred_tags = convert_id2tag(logits, id2tag)
            label_tags = convert_id2tag(labels.cpu().numpy().tolist(), id2tag)

            print('texts',texts)
            print('predict', pred_tags)
            print('labels', label_tags)
            predict_all.extend(pred_tags)
            labels_all.extend(label_tags)



    acc = accuracy_score(labels_all, predict_all)
    F1 = f1_score(labels_all, predict_all)
    P = precision_score(labels_all, predict_all)
    R = recall_score(labels_all, predict_all)
    report = classification_report(labels_all, predict_all)
    return loss_total / len(data_iter), acc, P, R, F1, report


def train(args, device):

    early_stop = 1024
    config = baseline_models.Config(args)
    embed = np.load('../static_data/embed.npy')
    # 这里要从 float64 转为 float32 不然会报错
    embed = embed.astype('float32')
    embed = torch.from_numpy(embed)
    with open('../static_data/vocab.json', 'r') as f:
        vocab = json.load(f)

    train_path = '../visual-data/train'
    dev_path = '../visual-data/dev'
    test_path = '../visual-data/test'

    train_inputs, train_labels = read_data(train_path,
                                           config.max_seq_length,
                                           config.tag2id)

    dev_inputs, dev_labels = read_data(dev_path,
                                       config.max_seq_length,
                                       config.tag2id)
    test_inputs, test_labels = read_data(test_path,
                                         config.max_seq_length,
                                         config.tag2id)

    train_indices, train_masks = trunc_and_padding(train_inputs, config, vocab)
    dev_indices, dev_masks = trunc_and_padding(dev_inputs, config, vocab)
    test_indices, test_masks = trunc_and_padding(test_inputs, config, vocab)

    train_labels = torch.LongTensor(train_labels)
    dev_labels = torch.LongTensor(dev_labels)
    test_labels = torch.LongTensor(test_labels)

    train_dataset = Data.TensorDataset(train_indices, train_labels, train_masks)
    dev_dataset = Data.TensorDataset(dev_indices, dev_labels, dev_masks)
    test_dataset = Data.TensorDataset(test_indices, test_labels, test_masks)

    train_iter = Data.DataLoader(train_dataset, config.batch_size)
    dev_iter = Data.DataLoader(dev_dataset, config.batch_size)
    test_iter = Data.DataLoader(test_dataset, config.batch_size)

    model = baseline_models.BaselineModel(args, embed, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    # loss = nn.CrossEntropyLoss()

    require_improvement = early_stop
    dev_best_loss = float('inf')
    # Record the iter of batch that the loss of the last validation set dropped
    last_improve = 0
    # Whether the result has not improved for a long time
    flag = False
    i = 0
    start = time.time()


    print(model)
    if args.test == 'train':
        for epoch in range(config.num_epochs):
            model.train()
            for X, y, masks in train_iter:
                X = X.to(device)
                y = y.to(device)
                masks = masks.to(device)

                loss = model(X, masks, y)
                optimizer.zero_grad()
                # l = loss(y_hat, y)
                loss.backward()
                optimizer.step()

                if (i + 1) % 100 == 0:
                    # pred = torch.max(y_hat.data.cpu(), 1)[1].numpy()
                    # acc = accuracy_score(y.detach().cpu().numpy(), pred)
                    # micro_F1 = f1_score(y.detach().cpu().numpy(), pred, average='micro')

                    dev_loss, dev_acc, dev_P, dev_R, dev_F1, dev_report = evaluate(model,
                                                                       dev_iter,
                                                                       device,
                                                                       'dev',
                                                                       config.id2tag)

                    model.train()

                    print('Epoch %d | iter %d | dev loss %f | dev accuracy %f | dev P %f | dev R %f | dev F1 %f' % (
                              epoch + 1, i + 1, dev_loss, dev_acc, dev_P, dev_R, dev_F1
                          ))

                    if dev_loss < dev_best_loss:
                        dev_best_loss = dev_loss
                        torch.save(model.state_dict(), config.save_path)
                        last_improve = i
                    model = model.to(device)

                if i - last_improve > require_improvement:
                    # Stop training if the loss of dev dataset has not dropped
                    # exceeds args.early_stop batches
                    print("No optimization for a long time, auto-stopping...")
                    flag = True
                    break
                i += 1
            if flag:
                break

        print('%.2f seconds used' % (time.time() - start))

        model = baseline_models.BaselineModel(args, embed, config).to(device)

        model.load_state_dict(torch.load(config.save_path))
        test_loss, test_acc, test_P, test_R, test_F1, test_report = evaluate(model,
                                                                test_iter,
                                                                device,
                                                                'test',
                                                                config.id2tag)
        print('test loss %f | test accuracy %f | test P %f | test R %f | test F1 %f' % (
                  test_loss, test_acc, test_P, test_R, test_F1))
        print("classification report: ")
        print(test_report)
    else:
        startTime = time.time()
        print('this test')
        model = baseline_models.BaselineModel(args, embed, config).to(device)

        model.load_state_dict(torch.load(config.save_path))

        test_loss, test_acc, test_P, test_R, test_F1, test_report = evaluate(model,
                                                                             test_iter,
                                                                             device,
                                                                             'test',
                                                                             config.id2tag)
        print('time:%.2f' % (time.time() - start))
        print('test loss %f | test accuracy %f | test P %f | test R %f | test F1 %f' % (
            test_loss, test_acc, test_P, test_R, test_F1))
        print("classification report: ")
        print(test_report)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(9)
torch.manual_seed(9)
torch.cuda.manual_seed_all(9)
torch.backends.cudnn.deterministic = True
parser = argparse.ArgumentParser()
parser.add_argument('-ms',
                    '--max_length',
                    type=int,
                    default=128)
parser.add_argument('-b',
                    '--batch_size',
                    type=int,
                    default=64)
parser.add_argument('-e',
                    '--num_epochs',
                    type=int,
                    default=100)
parser.add_argument('-es',
                    '--early_stop',
                    type=int,
                    default=1024)
parser.add_argument('-m',
                    '--model',
                    type=str,
                    help='LSTM, BiLSTM, GRU, BiGRU, LSTM_ATT, BiLSTM_ATT, GRU_ATT, BiGRU_ATT',
                    default='BiLSTM')
parser.add_argument('-t',
                    '--test',
                    type=str,
                    default='train')
args = parser.parse_args()
train(args, device)


