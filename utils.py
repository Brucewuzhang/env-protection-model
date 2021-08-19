import tensorflow as tf
from transformers import AutoTokenizer
import re

prompt = "环保或者环境保护？"


def example_gen(filepath, model_name, add_label=True):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def gen():
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                line = re.sub('.+：', '', line)
                if add_label:
                    title, label = line.split('\t')
                    inputs = tokenizer(prompt, title)
                    inputs = dict(inputs)
                    label = int(label)
                    inputs['labels'] = label
                else:
                    title = line
                    inputs = tokenizer(prompt, title)
                    inputs = dict(inputs)

                yield inputs

    return gen


def generate_data(filepath, model_name, batch_size, shuffle=True, add_label=True):
    data_gen = example_gen(filepath, model_name, add_label=add_label)
    output_signature = {"input_ids": tf.TensorSpec(shape=[None], dtype=tf.int32),
                        'token_type_ids': tf.TensorSpec(shape=[None], dtype=tf.int32),
                        "attention_mask": tf.TensorSpec(shape=[None], dtype=tf.int32)}
    padded_shapes = {"input_ids": [None],
                     'token_type_ids': [None],
                     "attention_mask": [None]}
    if add_label:
        output_signature['labels'] = tf.TensorSpec(shape=(), dtype=tf.int32)
        padded_shapes['labels'] = ()
    dataset = tf.data.Dataset.from_generator(data_gen,
                                             output_signature=output_signature)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)

    dataset = dataset.padded_batch(batch_size=batch_size, padded_shapes=padded_shapes)

    return dataset


if __name__ == "__main__":
    filename = 'env_train.txt'
    gen = example_gen(filename, 'hfl/chinese-electra-small-ex-discriminator')
    dataset = generate_data(filename, 'hfl/chinese-electra-small-ex-discriminator', batch_size=128)
    for e in dataset:
        print(e)
