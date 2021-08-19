import tensorflow as tf
from transformers import AutoTokenizer


def example_gen(filepath, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def gen():
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                title, label = line.split('\t')
                label = int(label)
                inputs = tokenizer(title)
                inputs = dict(inputs)
                inputs['labels'] = label
                yield inputs

    return gen


def generate_data(filepath, model_name, batch_size, shuffle=True):
    data_gen = example_gen(filepath, model_name)
    dataset = tf.data.Dataset.from_generator(data_gen,
                                             output_signature={
                                                 "input_ids": tf.TensorSpec(shape=[None], dtype=tf.int32),
                                                 'token_type_ids': tf.TensorSpec(shape=[None], dtype=tf.int32),
                                                 "attention_mask": tf.TensorSpec(shape=[None], dtype=tf.int32),
                                                 "labels": tf.TensorSpec(shape=(), dtype=tf.int32)})

    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)

    dataset = dataset.padded_batch(batch_size=batch_size, padded_shapes={"input_ids": [None],
                                                                         'token_type_ids': [None],
                                                                         "attention_mask": [None],
                                                                         "labels": ()})

    return dataset


if __name__ == "__main__":
    filename = 'env_train.txt'
    gen = example_gen(filename, 'hfl/chinese-electra-small-ex-discriminator')
    dataset = generate_data(filename, 'hfl/chinese-electra-small-ex-discriminator', batch_size=128)
    for e in dataset:
        print(e)
