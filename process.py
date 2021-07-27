import os
os.environ['TF_KERAS'] = '1'

from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding

from config import config


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
