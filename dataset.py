import os
os.environ['TF_KERAS'] = '1'

from bert4keras.snippets import sequence_padding


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