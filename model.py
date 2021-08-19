import tensorflow as tf
from transformers import TFElectraForSequenceClassification


class Classifier(tf.keras.Model):
    def __init__(self, model_name='hfl/chinese-electra-small-ex-discriminator', num_labels=2):
        super(Classifier, self).__init__()
        self.bone = TFElectraForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.accuracy_func = tf.keras.metrics.SparseCategoricalAccuracy()

    def call(self, inputs, training=None, mask=None):
        output = self.bone(inputs, training=training)
        labels = inputs.get("labels", None)
        if labels is not None:
            loss, logits = output[:2]
            self.add_loss(loss)
            accuracy = self.accuracy_func(labels, logits)
            self.add_metric(accuracy, "accuracy")
        else:
            logits = output[0]

        return logits




