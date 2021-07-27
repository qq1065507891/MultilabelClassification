import time
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle

from datetime import timedelta
from sklearn.model_selection import train_test_split


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


def name_to_dict(token_ids, segment_ids, labels):
    return {
               'Input-Token': token_ids,
               'Input-Segment': segment_ids
           }, labels


def pickle_data(contents, path):
    if not os.path.exists(path):
        with open(path, 'wb') as f:
            pickle.dump(contents, f)


def load_data(path):
    with open(path, 'rb') as f:
        contents = pickle.load(f)
    return contents


def read_file(path):
    return pd.read_csv(path)


def cut_train_dev(df):
    train_data, dev_data, train_label, dev_label = train_test_split(df['comment_text'],
                                                                    df[["toxic", "severe_toxic", "obscene",
                                                                        "threat", "insult", "identity_hate"]],
                                                                    random_state=15, test_size=0.3)
    return train_data, dev_data, train_label, dev_label


def predict_pd(datas):
    return datas['comment_text']