import time
import os

from utils import read_file, predict_pd, get_time_idf, pickle_data, load_data
from process import process_text_dataset
from train import load_model
from config import config


def predict_text_generator(config):
    start_time = time.time()
    if os.path.exists(config['test_pkl']):
        print('加载已有数据')
        text = load_data(config['test_pkl'])
    else:
        data = read_file(config['test_path'])
        data = predict_pd(data)
        print('处理数据')
        text = process_text_dataset(data, train=False)
        pickle_data(text, config['test_pkl'])
    model = load_model(config)
    pre = model.predict(text)
    pre = [1 if p > 0.5 else 0 for predict in pre for p in predict]
    end_time = get_time_idf(start_time)
    print('用时: ', end_time)
    return pre


def predict_single_text(text):
    text = process_text_dataset(text, train=False, generator=False)
    model = load_model(config)
    pre = model.predict(text)
    pre = [1 if p > 0.5 else 0 for p in pre[0]]
    return pre


if __name__ == '__main__':
    pre = predict_text_generator(config)
    print(pre)