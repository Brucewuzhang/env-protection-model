import argparse
import tensorflow as tf

from utils import generate_data
from model import Classifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='file to be predicted', required=True)
    parser.add_argument('--model_dir', type=str, help='saved model dir', required=True)
    parser.add_argument('--model_name', type=str, help='huggingface model name',
                        default='hfl/chinese-electra-small-ex-discriminator')
    parser.add_argument('--output', type=str, help='output file to write prediction label', required=True)

    args = parser.parse_args()
    datafile = args.input
    model_dir = args.model_dir
    model_name = args.model_name
    output = args.output

    dataset = generate_data(datafile, model_name, batch_size=256, shuffle=False, add_label=False)

    model = Classifier(model_name)
    latest_ckpt = tf.train.latest_checkpoint(model_dir)
    if not latest_ckpt:
        raise Exception("ckpt not found in {}".format(model_dir))

    model.load_weights(latest_ckpt).expect_partial()

    with open(output, 'w', encoding='utf-8') as f:
        label_list = []
        for e in dataset:
            preds = model(e, training=False)
            labels = preds.numpy().tolist()
            label_list.extend(labels)

        for label in label_list:
            f.write(str(label) + '\n')

    print("There are {} positive predictions".format(sum(label_list)))


if __name__ == "__main__":
    main()
