import tensorflow as tf
import argparse
import os

from model import Classifier
from utils import generate_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='input file', default='env_train.txt')
    parser.add_argument('--model_dir', type=str, help="dir to save trained model", required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_name', type=str, help='hugging face model name', default='hfl/chinese-electra-small-ex-discriminator')

    args = parser.parse_args()
    datafile = args.input
    model_dir = args.model_dir
    batch_size = args.batch_size
    model_name = args.model_name

    dataset = generate_data(datafile, model_name, batch_size)

    model = Classifier(model_name, num_labels=2)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer)

    os.makedirs(model_dir, exist_ok=True)
    latest_ckpt = tf.train.latest_checkpoint(model_dir)
    if latest_ckpt:
        model.load_weights(latest_ckpt).expect_partial()

    ckpt_path = 'model.ckpt-{epoch}'
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(os.path.join(model_dir, ckpt_path), verbose=1,
                                                       save_best_only=False, save_weights_only=True)

    model.fit(dataset, epochs=10, verbose=1, callbacks=[ckpt_callback])


if __name__ == "__main__":
    main()关键是建立现代环境保护制度	1
王利明：《物权法》与环境保护	1
蔡守秋：从环境权到国家环境保护义务和环境公益诉讼	1
