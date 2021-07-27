import os
import time
import tensorflow as tf

os.environ['TF_KERAS'] = '1'

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from process import read_file, cut_train_dev, process_text, pickle_data, load_data, get_time_idf, build_generator
from process import training_curve, process_text_dataset, name_to_dict
from config import config
from model import MultillabelClassification


def init_model(config):
    model = MultillabelClassification(config, last_activation='sigmoid', dropout_rate=0.3).build_model()

    model.compile(
        loss=tf.keras.losses.binary_crossentropy,
        optimizer=Adam(config['learning_rate']),
        metrics=['binary_accuracy'],
    )
    return model


def load_model(config):
    model = init_model(config)
    model.load_weights(config['model_file'])
    return model


def train_model(config):
    print('开始加载数据')
    stat_time = time.time()
    if os.path.exists(config['all_data_pkl']):
        all_data = load_data(config)
        train = all_data['train']
        dev = all_data['dev']
    else:
        df = read_file(config['ori_train_path'])
        train_data, dev_data, train_label, dev_label = cut_train_dev(df)
        # train = process_text(train_data, train_label)
        #
        # dev = process_text(dev_data, dev_label)

        train = process_text_dataset(train_data, train_label)

        dev = process_text_dataset(dev_data, dev_label)
        all_data = {
            'train': train,
            'dev': dev
        }
        pickle_data(all_data, config)

    # train_iter = build_generator(config, train)
    # dev_iter = build_generator(config, dev)
    train_iter = tf.data.Dataset.from_tensor_slices(train).map(name_to_dict).shuffle(buffer_size=100).batch(config['batch_size'])
    dev_iter = tf.data.Dataset.from_tensor_slices(dev).map(name_to_dict).batch(config['batch_size'])

    end_time = get_time_idf(stat_time)
    print('数据加载完成, 用时:{}, 训练数据:{}, 验证数据{}'.format(end_time, len(list(train_iter)), len(list(dev_iter))))

    if os.path.exists(config['model_file']):
        model = load_model(config)
        print('加载已有模型')
    else:
        model = init_model(config)
        print('模型初始化')

    cal_backs = [
        EarlyStopping(monitor='val_binary_accuracy', patience=10, verbose=1, mode='max'),
        ModelCheckpoint(config['model_file'], monitor='val_binary_accuracy', verbose=1,
                        save_weights_only=True, save_best_only=True, mode='max',
                        period=1)
    ]
    print('开始训练')
    start_time = time.time()
    history = model.fit(
        train_iter,
        epochs=config['epochs'],
        steps_per_epoch=len(list(train_iter)) // config['batch_size'],
        validation_data=dev_iter,
        validation_steps=len(list(dev_iter)) // config['batch_size'],
        callbacks=cal_backs,
        verbose=1
    )
    end_time = get_time_idf(start_time)
    print('训练结束,耗时: ', end_time)
    training_curve(history.history['loss'], history.history['binary_accuracy'],
                   history.history['val_loss'], history.history['binary_accuracy'])

    return history


if __name__ == '__main__':
    history = train_model(config)
    # test = r"Bye! Don't look, come or think of comming back! Tosser."
    # text = process_text(test, train=False, generator=False)
    # model = load_model(config)
    # # text_iter = build_generator(config, text)
    # pre = model.predict(text)
    # print(pre)
    # pre = [1 if p > 0.4 else 0 for p in pre[0]]
    # print(pre)
