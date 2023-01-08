import os
import json
import random
from collections import defaultdict
from tqdm import tqdm, trange
import datetime

import numpy as np
import torch
# from utils import tprint

from transformers import BertForMaskedLM, RobertaForMaskedLM, \
    BertConfig, BertTokenizer, RobertaConfig, RobertaTokenizer, \
    AlbertForMaskedLM, AlbertTokenizer, AlbertConfig




TASK_CLASSES = {
    'reuters': {
        'prompt_length': [1, 1, 1],
        'prompt': [lambda text: [text] + [',', 'the', 'topic', 'is', '[MASK]', '.'],
                   lambda text: [text] + ['.', 'What', 'is', 'the', 'topic', '?', '[MASK]', '.'],
                   lambda text: ['The', 'topic', ':', '[MASK]', '.', 'Input', ':'] + [text]],
        'block_flag': [[0, 0, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0, 0, 0, 0],
                       [1, 0, 0, 0, 0, 0, 0, 0]],
    },
    'huffpost': {
        'prompt_length': [1, 1, 1],
        'prompt': [lambda text: [text] + [',', 'the', 'topic', 'is', '[MASK]', '.'],
                   lambda text: [text] + ['.', 'What', 'is', 'the', 'topic', '?', '[MASK]', '.'],
                   lambda text: ['The', 'topic', ':', '[MASK]', '.', 'Input', ':'] + [text]],
        'block_flag': [[0, 0, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0, 0, 0, 0],
                       [1, 0, 0, 0, 0, 0, 0, 0]],
    },
    'amazon':{
        'prompt_length': [1, 1, 1],
        'prompt': [lambda text: [text] + [',', 'the', 'product', 'category', 'is', '[MASK]', '.'],
                   lambda text: [text] + ['What', 'is', 'the', 'product', 'category', '?', '[MASK]', '.'],
                   lambda text: ['The', 'product', 'category', ':', '[MASK]', '.', 'Input', ':'] + [text]],
        'block_flag': [[0, 0, 1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 1, 0, 0, 0, 0, 0],
                       [1, 0, 0, 0, 0, 0, 0, 0, 0]],
    },
    '20newsgroup':{
        'prompt_length': [1, 1, 1],
        'prompt': [lambda text: [text] + [',', 'the', 'topic', 'is', '[MASK]', '.'],
                   lambda text: [text] + ['.', 'What', 'is', 'the', 'topic', '?', '[MASK]', '.'],
                   lambda text: ['The', 'topic', ':', '[MASK]', '.', 'Input', ':'] + [text]],
        'block_flag': [[0, 0, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0, 0, 0, 0],
                       [1, 0, 0, 0, 0, 0, 0, 0]],
    },
}


MODEL_CLASSES = {
    'bert': {
        'config': BertConfig,
        'tokenizer': BertTokenizer,
        'mlm': BertForMaskedLM
    },
}


def tprint(s):
    '''
        print datetime and s
        @params:
            s (str): the string to be printed
    '''
    print('{}: {}'.format(
        datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S'), s),
          flush=True)


def _get_20newsgroup_classes():
    # label_dict = {'sci.space': 'space nasa orbit shuttle launch moon earth mission spacecraft',
    #               'sci.crypt': 'crypt encryption clip key keys chip nsa privacy government security algorithm',
    #               'sci.med': 'disease medical doctor food patients patient treatment medicine',
    #               'sci.electronics': 'electronics electronic power circuit amp phone radio chip current battery',
    #               'rec.autos': 'car cars engine auto speed',
    #               'rec.sport.hockey': 'hockey nhl espn game games team goal season sport',
    #               'rec.motorcycles': 'bike bikes biker bmw motorcycle motorcycles rider riding',
    #               'rec.sport.baseball': 'baseball game games team pitcher hitter braves league sport',
    #               'comp.graphics': 'computer graphics image images 3d bit color program software',
    #               'comp.windows.x': 'window windows motif file server program color application',
    #               'comp.os.ms-windows.misc': 'windows microsoft dos mouse driver system other',
    #               'comp.sys.mac.hardware': 'mac apple computer system hardware power drive monitor',
    #               'comp.sys.ibm.pc.hardware': 'computer system ibm pc hardware disk card dos controller monitor',
    #               'talk.politics.mideast': 'mideast armenian armenians turkish jew jews jewish israel israeli muslim arab',
    #               'misc.forsale': 'mail offer shipping price drive sell email interested system software computer games',
    #               'talk.politics.misc': 'politics government president gay homosexual cramer clinton state',
    #               'talk.politics.guns': 'gun guns fbi fire weapon government firearms law police',
    #               'talk.religion.misc': 'other god gods jesus bible christian religion morality biblical jews',
    #               'alt.atheism': 'god atheist morality moral islam islamic christian religion genocide',
    #               'soc.religion.christian': 'god jesus christian christianity christians christ bible faith hell heaven homosexuality doctrine'}

    # Answer words are selected based on label names and words appearing frequently in sample text
    train_classes = ['space nasa orbit shuttle launch moon earth mission spacecraft',
                     'crypt encryption clip key keys chip nsa privacy government security algorithm',
                     'disease medical doctor food patients patient treatment medicine',
                     'electronics electronic power circuit amp phone radio chip current battery',
                     'car cars engine auto speed',
                     'hockey nhl espn game games team goal season sport',
                     'bike bikes biker bmw motorcycle motorcycles rider riding',
                     'baseball game games team pitcher hitter braves league sport']
    val_classes = ['computer graphics image images 3d bit color program software',
                   'window windows motif file server program color application',
                   'windows microsoft dos mouse driver system other',
                   'mac apple computer system hardware power drive monitor',
                   'computer system ibm pc hardware disk card dos controller monitor']
    test_classes = ['mideast armenian armenians turkish jew jews jewish israel israeli muslim arab',
                    'mail offer shipping price drive sell email interested system software computer games',
                    'politics government president gay homosexual cramer clinton state',
                    'gun guns fbi fire weapon government firearms law police',
                    'other god gods jesus bible christian religion morality biblical jews',
                    'god atheist morality moral islam islamic christian religion genocide',
                    'god jesus christian christianity christians christ bible faith hell heaven homosexuality doctrine']

    return train_classes, val_classes, test_classes


def _get_amazon_classes():

    train_classes = ['Automotive', 'Baby', 'Beauty', 'Phones and Accessories',
                     'Grocery and Food', 'Health Care',
                     'Home and Kitchen', 'Patio and Lawn and Garden', 'Pet Supplies', 'Sports and Outdoors']
    val_classes = ['Android Apps', 'Toys and Games', 'Video Games', 'CDs', 'Digital Music']
    test_classes = ['Instant Video', 'Books', 'Kindle Store', 'Movies and TV', 'Clothing and Shoes and Jewelry',
                    'Electronics', 'Musical Instruments', 'Office Products', 'Tools']

    return train_classes, val_classes, test_classes


def _get_huffpost_classes():

    train_classes = ['POLITICS', 'WELLNESS', 'ENTERTAINMENT', 'TRAVEL',
                     'STYLE & BEAUTY', 'PARENTING', 'HEALTHY LIVING',
                     'QUEER VOICES', 'FOOD & DRINK', 'BUSINESS', 'COMEDY',
                     'SPORTS', 'BLACK VOICES', 'HOME & LIVING', 'PARENTS',
                     'THE WORLDPOST', 'WEDDINGS', 'WOMEN', 'IMPACT', 'DIVORCE']
    val_classes = ['CRIME', 'MEDIA', 'WEIRD NEWS', 'GREEN', 'WORLDPOST']
    test_classes = ['RELIGION', 'STYLE', 'SCIENCE', 'WORLD NEWS', 'TASTE',
                    'TECH', 'MONEY', 'ARTS', 'FIFTY', 'GOOD NEWS', 'ARTS & CULTURE',
                    'ENVIRONMENT', 'COLLEGE', 'LATINO VOICES', 'EDUCATION']

    return train_classes, val_classes, test_classes


def _get_reuters_classes():

    Reuters = [
        "acquisitions",
        "aluminium",
        "balance of payment",
        "cocoa",
        "coffee",
        "copper",
        "cotton",
        "consumer price index",
        "crude",
        "earn",
        "gross national product",
        "gold",
        "grain",
        "interest",
        "industrial production index",
        "iron and steel",
        "jobs",
        "livestock",
        "money foreign exchange",
        "money supply",
        "natural gas",
        "orange",
        "reserves",
        "retail",
        "rubber",
        "ship",
        "sugar",
        "tin",
        "trade",
        "vegetable oil",
        "wholesale price index"
    ]
    train_classes = Reuters[:15]
    val_classes = Reuters[15:20]
    test_classes = Reuters[20:]

    return train_classes, val_classes, test_classes


def drop_replicated_symbols(text):
    buffer = 0
    i = 0
    while i < len(text):
        if text[i] != text[i-1]:
            if buffer > 4:
                text = text[:i-buffer] + text[i:]
                i -= buffer
            buffer = 0
        else:
            buffer += 1
        i += 1
    if buffer > 0:
        text = text[:-buffer]
    return text


def _load_json(path, pretrained_model, model_type, cache_dir, dataset):

    label = {}
    text_len = []
    tokenizer = MODEL_CLASSES[pretrained_model]['tokenizer'].from_pretrained(
        model_type, cache_dir=cache_dir, do_lower_case=True)

    with open(path, 'r', errors='ignore') as f:
        data = []
        for line in f:
            row = json.loads(line)
            text_key = 'raw' if dataset == 'reuters' else 'text'
            if dataset == '20newsgroup' and 'Subject:' in row[text_key]:
                raw = row[text_key][row[text_key].find("Subject:"):]
            else:
                raw = row[text_key]

            # raw = raw.split(' ')[:500]
            # raw = ' '.join(raw)

            raw = tokenizer.tokenize(raw)
            raw = drop_replicated_symbols(raw)
            raw = raw[:500]

            albert_id = tokenizer.encode(raw)
            # filter out document with only special tokens
            # unk (100), cls (101), sep (102), pad (0)
            if np.max(albert_id) < 5:
                continue

            row_label = row['label']

            # count the number of examples per label
            if row_label not in label.keys():
                label[row_label] = 1
            else:
                label[row_label] += 1

            item = {
                'label': row_label,
                'raw': raw,
                'text_len': len(raw),
            }

            text_len.append(len(raw))
            data.append(item)

        tprint('Class balance:')
        print(label)
        tprint('Avg len: {}'.format(sum(text_len) / (len(text_len))))

    return data, label


def _meta_split(all_data, train_classes, val_classes, test_classes):
    train_data, val_data, test_data = [], [], []

    for example in all_data:
        if example['label'] in train_classes:
            train_data.append(example)
        if example['label'] in val_classes:
            val_data.append(example)
        if example['label'] in test_classes:
            test_data.append(example)

    return train_data, val_data, test_data


def _del_by_idx(array_list, idx):
    # modified to perform operations in place
    for i, array in enumerate(array_list):
        for remove_id in idx:
            array.remove(array[remove_id])
        array_list[i] = array

    if len(array_list) == 1:
        return array_list[0]
    else:
        return array_list


def _split_dataset(data, finetune_split):
    """
        split the data into train and val (maintain the balance between classes)
        @return data_train, data_val
    """
    # separate train and val data
    # used for fine tune
    data_train, data_val = defaultdict(list), defaultdict(list)

    # sort each matrix by ascending label order for each searching
    idx = np.argsort(data['label'], kind="stable")

    non_idx_keys = ['vocab_size', 'classes2id', 'is_train']
    for k, v in data.items():
        if k not in non_idx_keys:
            data[k] = v[idx]

    # loop through classes in ascending order
    classes, counts = np.unique(data['label'], return_counts=True)
    start = 0
    for label, n in zip(classes, counts):
        mid = start + int(finetune_split * n)  # split between train/val
        end = start + n  # split between this/next class

        for k, v in data.items():
            if k not in non_idx_keys:
                data_train[k].append(v[start:mid])
                data_val[k].append(v[mid:end])

        start = end  # advance to next class

    # convert back to np arrays
    for k, v in data.items():
        if k not in non_idx_keys:
            data_train[k] = np.concatenate(data_train[k], axis=0)
            data_val[k] = np.concatenate(data_val[k], axis=0)

    return data_train, data_val


def set_seed(seed):
    """
        Setting random seeds
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def load_dataset(args):
    set_seed(args.seed)

    tprint('Loading data')
    all_data, label = _load_json(args.data_path, args.pretrained_model, args.model_type,
                                 args.pretrained_cache_dir, args.dataset)

    if args.dataset == '20newsgroup':
        train_classes, val_classes, test_classes = _get_20newsgroup_classes()
    elif args.dataset == 'amazon':
        train_classes, val_classes, test_classes = _get_amazon_classes()
    elif args.dataset == 'huffpost':
        train_classes, val_classes, test_classes = _get_huffpost_classes()
    elif args.dataset == 'reuters':
        train_classes, val_classes, test_classes = _get_reuters_classes()
    else:
        raise ValueError(
            'args.dataset should be one of'
            '[20newsgroup, amazon, metatuning, huffpost, reuters]')

    # Split into meta-train, meta-val, meta-test data
    train_data, val_data, test_data = _meta_split(
        all_data, train_classes, val_classes, test_classes)
    tprint('#train {}, #val {}, #test {}'.format(
        len(train_data), len(val_data), len(test_data)))

    return train_data, val_data, test_data, train_classes, val_classes, test_classes


def build_episodes(
        data: list = None,
        n_way: int = 5,
        k_shot: int = 5,
        l_query: int = 25,
        n_episodes: int = 10,
        stored_episodes: str = ''
) -> list:
    """ build episode data from normal data """
    if stored_episodes and os.path.exists(stored_episodes):  # load pre-generated episodes
        with open(stored_episodes) as fp:
            episodes = json.load(fp)
    else:
        label_sample_map = create_label_sample_map(data)
        episodes = sample_episodes(label_sample_map,
                                   n_episodes=n_episodes,
                                   n_way=n_way, k_shot=k_shot, l_query=l_query)
        # episodes.append(episode)
        if stored_episodes:
            os.makedirs(stored_episodes[:stored_episodes.rfind('/')], exist_ok=True)
            with open(stored_episodes, 'w') as fp:
                json.dump(episodes, fp)
    return episodes


def create_label_sample_map(data):
    label_sample_map = {}
    for item in data:
        if item['label'] not in label_sample_map.keys():
            label_sample_map[item['label']] = [item]
        else:
            label_sample_map[item['label']].append(item)
    return label_sample_map


def sample_episodes(label_sample_map, n_episodes, n_way, k_shot, l_query, seed=None):
    if seed:
        np.random.seed(seed)

    episodes = []

    n_samples_per_label = min(len(label_sample_map[key]) for key, _ in label_sample_map.items())
    n_samples_total = len(label_sample_map) * n_samples_per_label
    n_samples_per_episode = n_way * (k_shot + l_query)
    n_sample_epochs = int(n_episodes * n_samples_per_episode / n_samples_total)+10
    for _ in trange(n_sample_epochs, desc='n_sample_epochs'):

        for label, items in label_sample_map.items():
            random.shuffle(items)

        keys = list(label_sample_map.keys())
        for sample_index in range(0, n_samples_per_label - n_samples_per_label%(k_shot+l_query), k_shot+l_query):
            random.shuffle(keys)

            for label_index in range(0, len(keys)-len(keys)%n_way, n_way):
                episode = {'support_set': [], 'query_set': []}
                episode['labels'] = keys[label_index: label_index+n_way]

                for i in range(label_index, label_index+n_way):
                    samples = label_sample_map[keys[i]][sample_index: sample_index + k_shot]
                    episode['support_set'] = episode['support_set']+samples
                    samples = label_sample_map[keys[i]][sample_index + k_shot: sample_index + k_shot + l_query]
                    episode['query_set'] = episode['query_set']+samples
                episodes.append(episode)
                if len(episodes) == n_episodes:
                    print(f"Generated {n_episodes} episodes")
                    return episodes

    return episodes

