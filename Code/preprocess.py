import os
import pyprind
import pandas as pd
from transformers import BertConfig, BertTokenizer


model_path = r'H:\huggingface\bert-base-chinese'
tokenizer_class = BertTokenizer
bert_config = BertConfig.from_pretrained(model_path, num_labels=2)
tokenizer = tokenizer_class.from_pretrained(model_path)

dir_names = ['part1', 'part2']
for dir_name in dir_names:
    dir_path = os.path.join(r'D:\pycharm\workspace\hzy\data', dir_name)
    file_names = os.listdir(dir_path)

    pper = pyprind.ProgPercent(len(file_names))
    for file_name in file_names:
        save_dir = os.path.join(r'D:\pycharm\workspace\hzy\data\labels', dir_name)
        save_path = os.path.join(save_dir, '%s.txt' % file_name.split('.')[0])

        df = pd.read_csv(os.path.join(dir_path, file_name), names=['label', 'content'])
        content = df['content'].tolist()
        label = df['label'].tolist()
        i = 0
        token_contents = []
        for c in content:
            if i == 0:
                i += 1
                token_contents.append(c)
            else:
                tokens = tokenizer.encode_plus(c, add_special_tokens=True, max_length=128,
                                               padding=True, truncation=True)['input_ids']
                token_content = []
                for token in tokens:
                    token_content.append(tokenizer._convert_id_to_token(token))
                token_contents.append(' '.join(token_content))
                i += 1

        with open(save_path, 'a', encoding='utf-8-sig') as f:
            for l, c in zip(label, token_contents):
                f.write('%s\t%s\n' % (l, c))

        pper.update()


# df = pd.read_csv('data/contents/content_1.pkl.csv', names=['label', 'content'])
# content = df['content'].tolist()
# label = df['label'].tolist()
# i = 0
# token_contents = []
# for c in content:
#     if i == 0:
#         i += 1
#         token_contents.append(c)
#     else:
#         tokens = tokenizer.encode_plus(c, add_special_tokens=True, max_length=128,
#                                        padding=True, truncation=True)['input_ids']
#         token_content = []
#         for token in tokens:
#             token_content.append(tokenizer._convert_id_to_token(token))
#         token_contents.append(' '.join(token_content))
#         i += 1
#
# with open('./data/label_1.txt', 'a', encoding='utf-8-sig') as f:
#     for l, c in zip(label, token_contents):
#         f.write('%s\t%s\n' % (l, c))


# df = pd.DataFrame({'label': label, 'content': token_contents})
# df.to_csv('./data/content.csv', index=False, header=False, encoding='utf-8-sig')
