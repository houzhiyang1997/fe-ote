import os
import re
import sys
import time
import json
import torch
import warnings
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
from transformers import BertTokenizer
from transformers import get_cosine_schedule_with_warmup
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

sys.path.append('..')
from models import interaction_gru


def get_label(line, comment_max_seq_length, tag2id, file_num, args):
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
    if args.mode == 'wbert' or args.mode == 'blstm':
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


def read_data(file_path, comment_max_seq_length, tag2id, args):
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
                weibo = contents[j]
            else:
                inputs.append('%s[SEP]%s' % (weibo, contents[j]))
                labels.append(get_label(lines[j], comment_max_seq_length,
                                        tag2id, file_num, args))

    return inputs, labels


def trunc_and_padding(texts, config, vocab, tokenizer):
    """
    截断与填充, 获得 mask, 并将 token 转为 id
    :param texts:
    :param config:
    :param vocab:
    :return:
    """
    input_weibos, input_comments = [], []
    for i in texts:
        weibo_and_comment = i.split('[SEP]')
        input_weibos.append(weibo_and_comment[0])
        input_comments.append(weibo_and_comment[1])

    weibo_indices, comments_indices, comment_masks, weibo_masks = [], [], [], []

    for text in input_weibos:
        tokens = tokenizer.encode_plus(text, add_special_tokens=True,
                                       max_length=config.comment_max_seq_length,
                                       padding='max_length', truncation=True)
        weibo_indices.append(tokens.input_ids)
        weibo_masks.append(tokens.attention_mask)

    for text in input_comments:
        words = list(text)
        ids = [vocab['PAD']] * config.comment_max_seq_length
        mask = [0] * config.comment_max_seq_length
        if len(words) > config.comment_max_seq_length:
            words = words[:config.comment_max_seq_length]
        i = 0
        for word in words:
            if vocab.__contains__(word):
                ids[i] = vocab[word]
            else:
                ids[i] = vocab['UNK']
            mask[i] = 1
            i += 1
        comments_indices.append(ids)
        comment_masks.append(mask)

    return torch.LongTensor(weibo_indices), torch.LongTensor(comments_indices),\
           torch.LongTensor(comment_masks), torch.LongTensor(weibo_masks)


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
        for weibos, comments, comment_masks, weibo_masks, input_labels in data_iter:

            # shape: (batch_size, seq_length)
            loss, pred = model(weibos,
                               comments,
                               input_labels,
                               mode,
                               input_masks=comment_masks,
                               input_weibo_masks=weibo_masks)

            # loss = F.cross_entropy(outputs, labels)

            loss_total += loss.data

            pred_tags = convert_id2tag(pred, id2tag)
            label_tags = convert_id2tag(input_labels.cpu().numpy().tolist(), id2tag)

            predict_all.extend(pred_tags)
            labels_all.extend(label_tags)

    acc = accuracy_score(labels_all, predict_all)
    p = precision_score(labels_all, predict_all)
    r = recall_score(labels_all, predict_all)
    f1 = f1_score(labels_all, predict_all)
    report = classification_report(labels_all,predict_all)
    return loss_total / len(data_iter), acc, p, r, f1, report


def train(args, device):
    config = interaction_gru.Config(args, num_outputs=3)

    train_path = '../data/train'
    dev_path = '../data/dev'
    test_path = '../data/test'

    train_inputs, train_labels = read_data(train_path,
                                           config.comment_max_seq_length,
                                           config.tag2id,
                                           args)
    dev_inputs, dev_labels = read_data(dev_path,
                                       config.comment_max_seq_length,
                                       config.tag2id,
                                       args)
    test_inputs, test_labels = read_data(test_path,
                                         config.comment_max_seq_length,
                                         config.tag2id,
                                         args)

    tokenizer_class, pretrained_weight = (BertTokenizer, config.model_path)
    tokenizer = tokenizer_class.from_pretrained(pretrained_weight)

    embed = np.load('../static_data/embed.npy')
    # 这里要从 float64 转为 float32 不然会报错
    embed = embed.astype('float32')
    embed = torch.from_numpy(embed)
    with open('../static_data/vocab.json', 'r') as f:
        vocab = json.load(f)

    train_weibos, train_comments, train_comment_masks, train_weibo_masks = trunc_and_padding(train_inputs,
                                                                                             config,
                                                                                             vocab,
                                                                                             tokenizer)
    dev_weibos, dev_comments, dev_comment_masks, dev_weibo_masks = trunc_and_padding(dev_inputs,
                                                                                     config,
                                                                                     vocab,
                                                                                     tokenizer)
    test_weibos, test_comments, test_comment_masks, test_weibo_masks = trunc_and_padding(test_inputs,
                                                                                         config,
                                                                                         vocab,
                                                                                         tokenizer
                                                                                         )
    train_labels = torch.LongTensor(train_labels)
    dev_labels = torch.LongTensor(dev_labels)
    test_labels = torch.LongTensor(test_labels)

    train_dataset = Data.TensorDataset(train_weibos,
                                       train_comments,
                                       train_comment_masks,
                                       train_weibo_masks,
                                       train_labels)
    dev_dataset = Data.TensorDataset(dev_weibos,
                                     dev_comments,
                                     dev_comment_masks,
                                     dev_weibo_masks,
                                     dev_labels)
    test_dataset = Data.TensorDataset(test_weibos,
                                      test_comments,
                                      test_comment_masks,
                                      test_weibo_masks,
                                      test_labels)

    train_iter = Data.DataLoader(train_dataset, config.batch_size)
    dev_iter = Data.DataLoader(dev_dataset, config.batch_size)
    test_iter = Data.DataLoader(test_dataset, config.batch_size)

    model = interaction_gru.Interaction(args, config, device, embed).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

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
        for weibos, comments, comment_masks, weibo_masks, input_labels in train_iter:
            optimizer.zero_grad()
            loss = model(weibos, comments, input_labels, mode='train',
                         input_masks=comment_masks, input_weibo_masks=weibo_masks)

            # l = loss(logits.permute(0, 2, 1), labels)
            loss.backward()
            optimizer.step()
            # schedule.step()

            if (n + 1) % 100 == 0:

                dev_loss, dev_acc, dev_P, dev_R, dev_f1, dev_report = evaluate(model,
                                                                   dev_iter,
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

    model = interaction_gru.Interaction(args, config, device, embed).to(device)
    model.load_state_dict(torch.load(config.save_path))
    test_loss, test_acc, test_P, test_R, test_f1, test_report = evaluate(model,
                                                            test_iter,
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
mode_selection = ['dot', 'minus_dot', 'minus', 'mlp', 'concat_mlp', 'mlp_concat']
modes = ['bbert', 'cbert', 'wbert', 'blstm']
parser = argparse.ArgumentParser()
parser.add_argument('-ws',
                    '--weibo_max_length',
                    type=int,
                    default=256)
parser.add_argument('-cs',
                    '--comment_max_length',
                    type=int,
                    default=128)
parser.add_argument('-b',
                    '--batch_size',
                    type=int,
                    default=4)
parser.add_argument('-e',
                    '--num_epochs',
                    type=int,
                    default=100)
parser.add_argument('-es',
                    '--early_stop',
                    type=int,
                    default=1024)
parser.add_argument('-m',
                    '--interaction_mode',
                    help='dot: dot product, '
                         'minus_dot: adding a minus sign after dot product, '
                         'minus: minus,'
                         'mlp: mlp',
                    default='dot')
parser.add_argument('--mode',
                    help='bbert: both using BERT to process weibos and comments, '
                         'cbert: using BERT to process comments and using BiLSTM to process weibos, '
                         'wbert: using BERT to process weibos and using BiLSTM to process comments, '
                         'blstm: both using BiLSTM to process weibos and comments.',
                    default='wbert')
args = parser.parse_args()
assert args.interaction_mode in mode_selection
assert args.mode in modes
train(args, device)

