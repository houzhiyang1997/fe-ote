import os
import re
import sys
import time
import torch
import warnings
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

sys.path.append('..')
from models import transformer


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
    text = line.split('\t')[1]

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
            if j == 0:
                continue
            else:
                inputs.append(contents[j])
                labels.append(get_label(lines[j], comment_max_seq_length, tag2id, file_num))

    return inputs, labels


def package_batch(inputs, labels, batch_size):
    """
    将数据封装成 batch
    :param inputs: list
            [weibo1[SEP]comment11, weibo1[SEP]comment12, weibo2[SEP]comment21, ...]
    :param labels: list
            只有目标词语的位置为 1, 其余位置为0
    :param batch_size: int
    :return:
    """
    assert len(inputs) == len(labels)

    if len(inputs) % batch_size != 0:
        flag = False
        batch_count = int(len(inputs) / batch_size) + 1
    else:
        flag = True
        batch_count = int(len(inputs) / batch_size)

    batch_X, batch_y = [], []

    if flag:
        for i in range(batch_count):
            batch_X.append(inputs[i * batch_size: (i + 1) * batch_size])
            batch_y.append(labels[i * batch_size: (i + 1) * batch_size])
    else:
        for i in range(batch_count):
            if i == batch_count - 1:
                batch_X.append(inputs[i * batch_size:])
                batch_y.append(labels[i * batch_size:])
            else:
                batch_X.append(inputs[i * batch_size: (i + 1) * batch_size])
                batch_y.append(labels[i * batch_size: (i + 1) * batch_size])

    return batch_X, batch_y, batch_count


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


def evaluate(model, batch_count, batch_X, batch_y, device, mode, id2tag):
    """
    评估模型
    :param model: Object
    :param batch_count: int
    :param batch_X: list
    :param batch_y: list
    :param device: Object
    :return:
    """
    model.eval()
    loss_total = 0
    predict_all = []
    labels_all = []

    with torch.no_grad():
        for i in range(batch_count):
            inputs = batch_X[i]
            labels = batch_y[i]

            # shape: (batch_size, seq_length)
            loss, pred = model(inputs,
                               torch.LongTensor(labels).to(device),
                               mode)

            # loss = F.cross_entropy(outputs, labels)

            loss_total += loss.data

            pred_tags = convert_id2tag(pred, id2tag)
            label_tags = convert_id2tag(labels, id2tag)

            predict_all.extend(pred_tags)
            labels_all.extend(label_tags)

    acc = accuracy_score(labels_all, predict_all)
    p = precision_score(labels_all, predict_all)
    r = recall_score(labels_all, predict_all)
    f1 = f1_score(labels_all, predict_all)
    report = classification_report(labels_all,predict_all)
    return loss_total / batch_count, acc, p, r, f1, report


def train(args, device):
    config = transformer.Config(args, num_outputs=3)

    train_path = '../data/train'
    dev_path = '../data/dev'
    test_path = '../data/test'

    train_inputs, train_labels = read_data(train_path,
                                           config.max_seq_length,
                                           config.tag2id)
    dev_inputs, dev_labels = read_data(dev_path,
                                       config.max_seq_length,
                                       config.tag2id)
    test_inputs, test_labels = read_data(test_path,
                                         config.max_seq_length,
                                         config.tag2id)
    train_batch_X, train_batch_y, train_batch_count = package_batch(train_inputs,
                                                                    train_labels,
                                                                    config.batch_size)
    dev_batch_X, dev_batch_y, dev_batch_count = package_batch(dev_inputs,
                                                              dev_labels,
                                                              config.batch_size)
    test_batch_X, test_batch_y, test_batch_count = package_batch(test_inputs,
                                                                 test_labels,
                                                                 config.batch_size)

    model = transformer.Transformer(args, config, device).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    # schedule = get_cosine_schedule_with_warmup(optimizer,
    #                                            num_warmup_steps=len(train_batch_X),
    #                                            num_training_steps=config.num_epochs * len(train_batch_X))

    early_stop = args.early_stop
    dev_best_loss = float('inf')
    # Record the iter of batch that the loss of the last validation set dropped
    last_improve = 0
    # Whether the result has not improved for a long time
    flag = False
    n = 0
    start = time.time()

    print('start training')
    for epoch in range(config.num_epochs):
        model.train()
        iter_start = time.time()
        for i in range(train_batch_count):
            inputs = train_batch_X[i]

            labels = torch.LongTensor(train_batch_y[i]).to(device)
            optimizer.zero_grad()

            # shape: (batch, seq_length, num_classes)
            loss = model(inputs, labels)
            # l = loss(logits.permute(0, 2, 1), labels)
            loss.backward()
            optimizer.step()
            # schedule.step()

            if (n + 1) % 100 == 0:

                dev_loss, dev_acc, dev_P, dev_R, dev_f1, dev_report = evaluate(model,
                                                                   dev_batch_count,
                                                                   dev_batch_X,
                                                                   dev_batch_y,
                                                                   device,
                                                                   'dev',
                                                                   config.id2tag)
                model.train()
                print('Epoch %d | Iter %d |  Dev loss %f | Dev acc %f | Dev macro P %f | '
                      'Dev macro R %f | Dev macro F1 %f | Duration %.2f' % (
                          epoch + 1, n + 1, dev_loss, dev_acc, dev_P,
                          dev_R, dev_f1, time.time() - iter_start
                      ))

                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    last_improve = n

            if n - last_improve > early_stop:
                # Stop training if the loss of dev dataset has not dropped
                # exceeds early_stop batches
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
            n += 1
        if flag:
            break

    print('%.2f seconds used' % (time.time() - start))

    model = transformer.Transformer(args, config, device).to(device)
    model.load_state_dict(torch.load(config.save_path))
    test_loss, test_acc, test_P, test_R, test_f1, test_report = evaluate(model,
                                                            test_batch_count,
                                                            test_batch_X,
                                                            test_batch_y,
                                                            device,
                                                            'test',
                                                            config.id2tag)
    print('Test loss %f | Test acc %f | Test macro P %f | Test macro R %f | Test macro F1 %f' % (
        test_loss, test_acc, test_P, test_R, test_f1
    ))
    print("classification report: ")
    print(test_report)


warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(9)
torch.manual_seed(9)
torch.cuda.manual_seed_all(9)
torch.backends.cudnn.deterministic = True
parser = argparse.ArgumentParser()
parser.add_argument('-cs',
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
parser.add_argument('-s',
                    '--is_seq',
                    default='')
args = parser.parse_args()
train(args, device)

