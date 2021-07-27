import pandas as pd
import pickle
import time
import matplotlib.pyplot as plt
import os
os.environ['TF_KERAS'] = '1'

from datetime import timedelta
from sklearn.model_selection import train_test_split
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding

from config import config


def read_file(path):
    return pd.read_csv(path)


def cut_train_dev(df):
    train_data, dev_data, train_label, dev_label = train_test_split(df['comment_text'],
                                                                    df[["toxic", "severe_toxic", "obscene",
                                                                        "threat", "insult", "identity_hate"]],
                                                                    random_state=15, test_size=0.3)
    return train_data, dev_data, train_label, dev_label


def process_text(texts, labels=None, train=True, generator=True):
    tokenizer = Tokenizer(config['dict_path'], do_lower_case=True)
    contents = []

    if not train and not generator:
        token_ids, segment_ids = tokenizer.encode(texts, maxlen=config['maxlen'])
        contents.append((token_ids, segment_ids))
    elif not train and generator:
        for text in texts.values:
            token_ids, segment_ids = tokenizer.encode(text, maxlen=config['maxlen'])
            contents.append((token_ids, segment_ids))
    else:
        for text, label in zip(texts.values, labels.values):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=config['maxlen'])
            contents.append((token_ids, segment_ids, label.tolist()))
    return contents


def process_text_dataset(texts, labels=None, train=True, generator=True):
    tokenizer = Tokenizer(config['dict_path'], do_lower_case=True)

    if not train and not generator:
        token_ids, segment_ids = tokenizer.encode(texts, maxlen=config['maxlen'])
        token_ids_pad = sequence_padding([token_ids])
        segment_ids_pad = sequence_padding([segment_ids])
        content = (token_ids_pad, segment_ids_pad)
    elif not train and generator:
        contents = []
        for text in texts.values:
            token_ids, segment_ids = tokenizer.encode(text, maxlen=config['maxlen'])
            contents.append((token_ids, segment_ids))
        token_ids_pad = sequence_padding([item[0] for item in contents])
        segment_ids_pad = sequence_padding([item[1] for item in contents])
        content = (token_ids_pad, segment_ids_pad)
    else:
        contents = []
        for text, label in zip(texts.values, labels.values):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=config['maxlen'])
            contents.append((token_ids, segment_ids, label.tolist()))
        token_ids_pad = sequence_padding([item[0] for item in contents])
        segment_ids_pad = sequence_padding([item[1] for item in contents])
        all_label_pad = sequence_padding([item[2] for item in contents])
        content = (token_ids_pad, segment_ids_pad, all_label_pad)
    return content


def name_to_dict(token_ids, segment_ids, labels):
    return {
        'Input-Token': token_ids,
        'Input-Segment': segment_ids
    }, labels


def pickle_data(contents, config):
    if not os.path.exists(config['all_data_pkl']):
        with open(config['all_data_pkl'], 'wb') as f:
            pickle.dump(contents, f)


def load_data(config):
    with open(config['all_data_pkl'], 'rb') as f:
        contents = pickle.load(f)
    return contents


def training_curve(loss, acc, val_loss=None, val_acc=None):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(loss, color='r', label='Training Loss')
    if val_loss is not None:
        ax[0].plot(val_loss, color='g', label='Validation Loss')
    ax[0].legend(loc='best', shadow=True)
    ax[0].grid(True)

    ax[1].plot(acc, color='r', label='Training Accuracy')
    if val_loss is not None:
        ax[1].plot(val_acc, color='g', label='Validation Accuracy')
    ax[1].legend(loc='best', shadow=True)
    ax[1].grid(True)
    plt.show()


def get_time_idf(start_time):
    """
    获取已经使用的时间
    :param start_time:
    :return: 返回使用多长时间
    """
    end_time = time.time()
    time_idf = end_time - start_time
    return timedelta(seconds=int(round(time_idf)))


class DataGenerator(object):
    """
        数据迭代器
    """
    def __init__(self, dataset, batch_size, train=True):
        self.dataset = dataset
        self.train = train
        self.batch_size = batch_size
        self.n_batches = len(dataset) // batch_size
        self.index = 0

    def _to_tensor(self, datas):
        token_ids = sequence_padding([item[0] for item in datas])
        segment_ids = sequence_padding([item[1] for item in datas])
        if not self.train:
            return [token_ids, segment_ids]
        else:
            y = sequence_padding([item[2] for item in datas])
            return [token_ids, segment_ids], y

    def __next__(self):
        if self.index == self.n_batches:
            batches = self.dataset[self.index * self.batch_size: len(self.dataset)]
            self.index = 0
            batches = self._to_tensor(batches)
            return batches
        else:
            batches = self.dataset[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index = self.index + 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        return self.n_batches + 1


def build_generator(config, dataset):
    return DataGenerator(dataset, config['batch_size'])
