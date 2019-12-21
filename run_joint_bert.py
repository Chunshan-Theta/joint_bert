# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import modeling
import optimization
import tokenization
import tensorflow as tf
import pickle
import json

from sklearn.utils import class_weight
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import metrics_impl
from tensorflow.python.ops import variable_scope
from tensorflow.python.distribute import distribution_strategy_context

import tf_metrics


data_dir = 'data/'
bert_config_file = 'bert_wwm/bert_config.json'
task_name = "customized"
output_dir = './output/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
init_checkpoint = 'bert_wwm/bert_model.ckpt'
do_lower_case = True
max_seq_length =64
use_tpu = False
use_one_hot_embeddings = False
do_train = True
do_eval = True
do_predict = True
train_batch_size = 16
eval_batch_size = 16
predict_batch_size = 16
learning_rate = 5e-5
num_train_epochs = 15
warmup_proportion = 0.1
save_checkpoints_steps = 1000
iterations_per_loop = 1000
vocab_file = './bert_wwm/vocab.txt'
master = None
num_tpu_cores = 8
crf_loss_method=True
do_export = False
export_dir = None

alpha = 0.01  # adjust weight between slot and intent, loss = alpha*(slot loss) + (1-alpha)*(intent loss)



####
class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, tags, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.tags = tags
        self.label = label


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 tags_ids,
                 label_ids,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.tags_ids = tags_ids
        self.label_ids = label_ids
        self.is_real_example = is_real_example


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

        @classmethod
        def _read_data(cls, input_file):
            """Reads a BIO data."""
            with open(input_file) as f:
                lines = []
                words = []
                labels = []
                for line in f:
                    contends = line.strip()
                    word = line.strip().split(' ')[0]
                    label = line.strip().split(' ')[-1]
                    if contends.startswith("-DOCSTART-"):
                        words.append('')
                        continue
                    # if len(contends) == 0 and words[-1] == '。':
                    if len(contends) == 0:
                        l = ' '.join([label for label in labels if len(label) > 0])
                        w = ' '.join([word for word in words if len(word) > 0])
                        lines.append([l, w])
                        words = []
                        labels = []
                        continue
                    words.append(word)
                    labels.append(label)
                return lines


class customizedProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            

            self._read_tsv(os.path.join(data_dir, "train-joint.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev-joint.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test-joint.tsv")), "test")

    def get_labels_info(self):
        labels = []
        tags = []
        label_map = {}
        tags_map = {}
        label_map_file = os.path.join(output_dir, "label_map.txt")
        tags_map_file = os.path.join(output_dir, "tags_map.txt")
        lines = self._read_tsv(os.path.join(data_dir, "train-joint.tsv"))

        for line in lines:
            line_of_labels = line[0].strip("[]").split(', ')
            for label in line_of_labels:
                labels.append(label)
            tags += line[2].strip().split()

        tags.append("[CLS]")
        # tags.append("[SEP]")

        all_labels = labels  # for cal intent_weights
        all_tags = tags  # for cal intent_weights

        labels = sorted(set(labels), reverse=False)
        tags = sorted(set(tags), reverse=False)
        num_labels = sorted(set(labels), reverse=True).__len__()
        num_tags = sorted(set(tags), reverse=True).__len__()

        intent_class_weights = class_weight.compute_class_weight('balanced',
                                                                 labels,
                                                                 all_labels)

        slot_class_weights = class_weight.compute_class_weight('balanced',
                                                               tags,
                                                               all_tags)

        with tf.gfile.GFile(label_map_file, "w") as writer:
            for (i, label) in enumerate(labels):
                label_map[label] = i
                writer.write("{}:{}\n".format(i, label))

        with tf.gfile.GFile(tags_map_file, "w") as writer:
            for (i, tag) in enumerate(tags):
                tags_map[tag] = i
                writer.write("{}:{}\n".format(i, tag))

        return label_map, num_labels, num_tags, tags_map, intent_class_weights, slot_class_weights

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            label = tokenization.convert_to_unicode(line[0])
            tags = tokenization.convert_to_unicode(line[2])
            text = tokenization.convert_to_unicode(line[1])
            examples.append(InputExample(guid=guid, text=text, tags=tags, label=label))
        return examples


def write_tokens(tokens, mode):
    if mode == "test":
        path = os.path.join(output_dir, "token_" + mode + ".txt")
        wf = open(path, 'a')
        for token in tokens:
            if token != "**NULL**":
                wf.write(token + '\n')
        wf.close()


def convert_single_example(ex_index, example, label_map, tags_map,
                           max_seq_length, tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            tags_ids=[0] * max_seq_length,
            label_ids=[0] * len(label_map),
            is_real_example=False)

    tokens_list = example.text
    tags_list = example.tags.strip().split(" ")
    tokens = []
    tags = []

    tokens.append("[CLS]")
    tags.append("[CLS]")

    for i, (word, tag) in enumerate(zip(tokens_list, tags_list)):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        for i, _ in enumerate(token):
            if i == 0:
                tags.append(tag)
            else:
                tags.append("O")

    if len(tokens) >= max_seq_length:
        tokens = tokens[0:max_seq_length]
        tags = tags[0:max_seq_length]

    # tokens.append("[SEP]")
    # tags.append("[SEP]")

    segment_ids = [0] * max_seq_length
    tags_ids = [tags_map[tag] for tag in tags]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    label_list = example.label.strip('[]').split(', ')
    multi_label_list = [0] * len(label_map)

    for label in label_list:
        label_index = label_map[label]
        multi_label_list[label_index] = 1

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        tags_ids.append(tags_map["O"])

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(tags_ids) == max_seq_length

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("label: %s" % (example.label))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info(
            "input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info(
            "input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info(
            "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("tags_ids: %s" % " ".join([str(x) for x in tags_ids]))
    feature = InputFeatures(
        label_ids=multi_label_list,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        tags_ids=tags_ids)
    return feature


def file_based_convert_examples_to_features(
        examples, label_map, tags_map, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""
    writer = tf.python_io.TFRecordWriter(output_file)
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info(
                "Writing example %d of %d" % (ex_index, len(examples)))
        feature = convert_single_example(ex_index, example, label_map,
                                         tags_map, max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["label_ids"] = create_int_feature(feature.label_ids)
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["tags_ids"] = create_int_feature(feature.tags_ids)
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])

        tf_example = tf.train.Example(
            features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder, num_labels):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "label_ids": tf.FixedLenFeature([num_labels], tf.int64),
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "tags_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 label_ids, tag_ids, intent_num_labels, num_tags, intent_class_weights, slot_class_weights,
                 use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    intent_output_layer = model.get_pooled_output()
    sequence_output_layer = model.get_sequence_output()

    hidden_size = sequence_output_layer.shape[-1].value

    ## intent loss
    intent_output_weights = tf.get_variable(
        "intent_output_weights", [intent_num_labels, hidden_size],
        # initializer=tf.truncated_normal_initializer(stddev=0.02)
        initializer=tf.contrib.layers.xavier_initializer()
    )

    intent_output_bias = tf.get_variable(
        "intent_output_bias", [intent_num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("intent_loss"):
        if is_training:
            # I.e., 0.1 dropout
            intent_output_layer = tf.nn.dropout(intent_output_layer, keep_prob=0.9)

        intent_logits = tf.matmul(intent_output_layer, intent_output_weights, transpose_b=True)
        intent_logits = tf.nn.bias_add(intent_logits, intent_output_bias, name='intent_logits')
        intent_probabilities = tf.nn.sigmoid(intent_logits, name='intent_probabilities')
        # classes_weights = tf.constant([1] * intent_num_labels, dtype=tf.float32)

        classes_weights = tf.constant(intent_class_weights, dtype=tf.float32)

        weighted_cross_entropy = tf.nn.weighted_cross_entropy_with_logits(labels=tf.cast(label_ids, tf.float32),
                                                                          logits=intent_logits,
                                                                          pos_weight=classes_weights)
        intent_per_example_loss = tf.reduce_sum(weighted_cross_entropy, axis=-1)

    # slot loss crf
    with tf.variable_scope("slot_loss"):
        if is_training:
            # I.e., 0.1 dropout
            sequence_output_layer = tf.nn.dropout(sequence_output_layer, keep_prob=0.9)

        sequence_logits = tf.layers.dense(
            inputs=sequence_output_layer,
            units=num_tags,
            use_bias=True,
            bias_initializer=tf.zeros_initializer(),
            # kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        mask_length = tf.reduce_sum(input_mask, axis=1)

    log_likelihood, transition = tf.contrib.crf.crf_log_likelihood(
        inputs=sequence_logits,
        tag_indices=tag_ids,
        sequence_lengths=mask_length)

    decode_tags, best_score = tf.contrib.crf.crf_decode(
        potentials=sequence_logits,
        transition_params=transition,
        sequence_length=mask_length)

    loss = alpha * tf.reduce_sum(-log_likelihood) + (1 - alpha) * tf.reduce_mean(
        intent_per_example_loss)

    return (loss, sequence_logits, intent_logits, decode_tags, intent_probabilities)


def f1_score(labels, predictions, weights=None, num_thresholds=200,
             metrics_collections=None, updates_collections=None, name=None):
    with variable_scope.variable_scope(
            name, 'f1', (labels, predictions, weights)):
        predictions, labels, weights = metrics_impl._remove_squeezable_dimensions(  # pylint: disable=protected-access
            predictions=predictions, labels=labels, weights=weights)
        # To account for floating point imprecisions / avoid division by zero.
        epsilon = 1e-7
        thresholds = [(i + 1) * 1.0 / (num_thresholds - 1)
                      for i in range(num_thresholds - 2)]
        thresholds = [0.0 - epsilon] + thresholds + [1.0 + epsilon]
        thresholds_tensor = tf.constant(thresholds)

        # Confusion matrix.
        values, update_ops = metrics_impl._confusion_matrix_at_thresholds(  # pylint: disable=protected-access
            labels, predictions, thresholds, weights, includes=('tp', 'fp', 'fn'))

        # Compute precision and recall at various thresholds.
        def compute_best_f1_score(tp, fp, fn, name):
            precision_at_t = math_ops.div(tp, epsilon + tp + fp,
                                          name='precision_' + name)
            recall_at_t = math_ops.div(tp, epsilon + tp + fn, name='recall_' + name)
            # Compute F1 score.
            f1_at_thresholds = (
                    2.0 * precision_at_t * recall_at_t /
                    (precision_at_t + recall_at_t + epsilon))

            best_f1 = math_ops.reduce_max(f1_at_thresholds)
            best_f1_index = tf.math.argmax(f1_at_thresholds)
            precision = precision_at_t[best_f1_index]
            recall = recall_at_t[best_f1_index]
            threshold = thresholds_tensor[best_f1_index]
            return best_f1, precision, recall, threshold

        def f1_across_replicas(_, values):
            best_f1, precision, recall, threshold = compute_best_f1_score(tp=values['tp'], fp=values['fp'],
                                                                          fn=values['fn'], name='value')
            if metrics_collections:
                ops.add_to_collections(metrics_collections, best_f1, precision, recall, threshold)
            return best_f1, precision, recall, threshold

        best_f1, precision, recall, threshold = distribution_strategy_context.get_replica_context().merge_call(
            f1_across_replicas, args=(values,))

        update_op = compute_best_f1_score(tp=update_ops['tp'], fp=update_ops['fp'],
                                          fn=update_ops['fn'], name='update')
        if updates_collections:
            ops.add_to_collections(updates_collections, update_op)

        # return (best_f1, precision, recall, threshold), update_op
        return (best_f1, update_op), (precision, update_op), (recall, update_op), (threshold, update_op)
        # return best_f1, precision, recall, threshold


def model_fn_builder(bert_config, num_labels, num_tags, intent_class_weights, slot_class_weights, init_checkpoint,
                     learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        tags_ids = features["tags_ids"]
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, sequence_logits, intent_logits, decode_tags, probabilities) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids,
            label_ids, tags_ids, num_labels, num_tags, intent_class_weights, slot_class_weights, use_one_hot_embeddings)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(tags_ids, label_ids, decode_tags, probabilities, num_tags, input_mask):
                slot_accuracy = tf.metrics.accuracy(tags_ids, decode_tags, input_mask)

                slot_precision = tf_metrics.precision(tags_ids, decode_tags, num_tags, weights=input_mask,
                                                      average="macro")
                slot_recall = tf_metrics.recall(tags_ids, decode_tags, num_tags, weights=input_mask, average="macro")
                slot_f1 = tf_metrics.f1(tags_ids, decode_tags, num_tags, weights=input_mask, average="macro")
                best_f1, precision, recall, threshold = f1_score(label_ids, probabilities)
                evl_metrics = {}
                evl_metrics.update({'class_intent_f1': best_f1})
                evl_metrics.update({'class_precision': precision})
                evl_metrics.update({'class_recall': recall})
                evl_metrics.update({'class_threshold_': threshold})
                for i in range(num_labels):
                    best_f1, precision, recall, threshold = f1_score(label_ids[:, i], probabilities[:, i])
                    evl_metrics[f'class{i:0>2d}_f1'] = best_f1
                    evl_metrics[f'class{i:0>2d}_precision'] = precision
                    evl_metrics[f'class{i:0>2d}_recall'] = recall
                    evl_metrics[f'class{i:0>2d}_threshold'] = threshold

                evl_metrics.update({'slot_accuracy': slot_accuracy,
                                    'slot_precision': slot_precision,
                                    'slot_recall': slot_recall,
                                    'slot_f1': slot_f1
                                    })

                for metric_name, op in evl_metrics.items():
                    tf.summary.scalar(metric_name, op[1])

                return evl_metrics

            eval_metrics = (metric_fn, [tags_ids, label_ids, decode_tags, probabilities, num_tags, input_mask])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={"slot_predicted_ids": decode_tags,
                             "intent_predicted": probabilities,
                             "tags_ids": tags_ids,
                             "label_ids": label_ids,
                             "input_ids": input_ids,
                             "mask_length": tf.reduce_sum(input_mask, axis=1)},
                scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_ids)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    all_segment_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "label_ids":
                tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
        })

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        features.append(feature)
    return features


def main():
    tf.logging.set_verbosity(tf.logging.INFO)

    processors = {
        "customized": customizedProcessor
    }

    tokenization.validate_case_matches_checkpoint( do_lower_case,
                                                   init_checkpoint)

    if not  do_train and not  do_eval and not  do_predict and not  do_export:
        raise ValueError(
            "At least one of `do_train`, `do_eval`, `do_predict' or 'do_export' must be True.")

    bert_config = modeling.BertConfig.from_json_file( bert_config_file)

    if  max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            ( max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs( output_dir)


    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    label_map, num_labels, num_tags, tags_map, intent_class_weights, slot_class_weights = processor.get_labels_info()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)

    tpu_cluster_resolver = None
    if use_tpu and tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            tpu_name, zone=tpu_zone, project=gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=master,
        model_dir=output_dir,
        save_checkpoints_steps=save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=iterations_per_loop,
            num_shards=num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if do_train:
        train_examples = processor.get_train_examples(data_dir)
        num_train_steps = int(
            len(train_examples) / train_batch_size * num_train_epochs)
        num_warmup_steps = int(num_train_steps * warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=num_labels,
        num_tags=num_tags,
        intent_class_weights=intent_class_weights,
        slot_class_weights=slot_class_weights,
        init_checkpoint=init_checkpoint,
        learning_rate=learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=use_tpu,
        use_one_hot_embeddings=use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        predict_batch_size=predict_batch_size)

    if do_train:
        train_file = os.path.join(output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, label_map, tags_map, max_seq_length, tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=max_seq_length,
            is_training=True,
            drop_remainder=False,
            num_labels=num_labels)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if do_eval:
        eval_examples = processor.get_dev_examples(data_dir)
        num_actual_eval_examples = len(eval_examples)

        if use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on. These do NOT count towards the metric (all tf.metrics
            # support a per-instance weight, and these get a weight of 0.0).
            while len(eval_examples) %FLAGS.eval_batch_size != 0:
                eval_examples.append(PaddingInputExample())

        eval_file = os.path.join(output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, label_map, tags_map, max_seq_length, tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if use_tpu:
            assert len(eval_examples) % eval_batch_size == 0
            eval_steps = int(len(eval_examples) // eval_batch_size)

        eval_drop_remainder = True if use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder,
            num_labels=num_labels)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(output_dir, "eval_results.txt")
        best_threshold_file = os.path.join(output_dir, "best_class_threshold.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            with tf.gfile.GFile(best_threshold_file, "w") as threshold_writer:
                tf.logging.info("***** Eval results *****")
                for key in result.keys():
                    tf.logging.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))
                    if key.endswith('threshold'):
                        threshold_writer.write("%s\n" % str(result[key]))
                        tf.logging.info("")
                        writer.write("\n")

    if do_predict:
        predict_examples = processor.get_test_examples(data_dir)
        num_actual_predict_examples = len(predict_examples)
        if use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on.
            while len(predict_examples) % predict_batch_size != 0:
                predict_examples.append(PaddingInputExample())

        predict_file = os.path.join(output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(predict_examples, label_map, tags_map,
                                                max_seq_length, tokenizer, predict_file)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", predict_batch_size)

        predict_drop_remainder = True if use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder,
            num_labels=num_labels)

        result = estimator.predict(input_fn=predict_input_fn)
        id2label = {v: k for k, v in label_map.items()}
        id2tag = {v: k for k, v in tags_map.items()}
        output_predict_file = os.path.join(output_dir, "test_results.tsv")

        best_threshold_file = os.path.join(output_dir, "best_class_threshold.txt")
        if not tf.gfile.Exists(best_threshold_file):
            threshold = [0.5] * num_labels
        else:
            with tf.gfile.GFile(best_threshold_file, "r") as reader:
                threshold = reader.read().splitlines()
            threshold = list(map(float, threshold))

        with tf.gfile.GFile(output_predict_file, "w") as writer:
            num_written_lines = 0
            output_line = "\t".join(
                ['frame_O_X', 'intent OX', 'intent_label', 'intent_pred', 'sentence', 'slot_pred', 'slot_label',
                 'slot OX']) + "\n"
            writer.write(output_line)

            # below for cal metrics
            slot_pred_all_data = []
            slot_label_all_data = []
            intent_pred_all_data = []
            intent_label_all_data = []

            for index, item in enumerate(result):
                mask_length = item["mask_length"]
                intent_predicted = item["intent_predicted"]
                intent_predicted = [intent_predicted[i] > threshold[i] for i in range(num_labels)]

                label_ids = item["label_ids"]
                tags_ids = item["tags_ids"][:mask_length]
                slot_predicted_ids = item["slot_predicted_ids"][:mask_length]
                input_ids = item["input_ids"][:mask_length]
                tokens = tokenizer.convert_ids_to_tokens(input_ids)

                slot_pred_all_data.extend(slot_predicted_ids)
                slot_label_all_data.extend(tags_ids)

                assert len(slot_predicted_ids) == len(tags_ids)

                intent_pred_all_data.append(intent_predicted)
                intent_label_all_data.append(label_ids)

                pre_intent = ' '.join([id2label[index] for index, pred in enumerate(intent_predicted) if pred])
                pre_tags = ' '.join([id2tag[pred_tag] for pred_tag in slot_predicted_ids])
                label_intent = ' '.join([id2label[index] for index, label in enumerate(label_ids) if label])
                label_tags = ' '.join([id2tag[label_tag] for label_tag in tags_ids])

                tokens = ''.join(tokens)
                slot_compare = 'O' if pre_tags == label_tags else 'X'
                intent_compare = 'O' if pre_intent == label_intent else 'X'
                frame_O_X = 'O' if pre_intent == label_intent and pre_tags == label_tags else 'X'
                output_line = "\t".join(
                    [frame_O_X, intent_compare, label_intent, pre_intent, tokens, pre_tags, label_tags,
                     slot_compare]) + "\n"
                writer.write(output_line)
                num_written_lines += 1
            assert num_written_lines == num_actual_predict_examples

        # write test metrics
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        import numpy as np
        output_test_metrics_file = os.path.join(output_dir, "test_results.txt")
        with tf.gfile.GFile(output_test_metrics_file, "w") as writer:

            intent_label_all_data = np.array(intent_label_all_data)
            intent_pred_all_data = np.array(intent_pred_all_data)

            for i in range(num_labels):
                f1 = f1_score(intent_label_all_data[:, i], intent_pred_all_data[:, i])
                precision = precision_score(intent_label_all_data[:, i], intent_pred_all_data[:, i])
                recall = recall_score(intent_label_all_data[:, i], intent_pred_all_data[:, i])

                writer.write(f'class{i:0>2d}_f1 = {f1}\n')
                writer.write(f'class{i:0>2d}_precision = {precision}\n')
                writer.write(f'class{i:0>2d}_recall = {recall}\n')
                writer.write('\n\n')

            intent_label_flatten = intent_label_all_data.reshape(-1)
            intent_pred_flatten = intent_pred_all_data.reshape(-1)
            intent_f1 = f1_score(intent_label_flatten, intent_pred_flatten)
            intent_precision = precision_score(intent_label_flatten, intent_pred_flatten)
            intent_recall = recall_score(intent_label_flatten, intent_pred_flatten)

            writer.write(f'class_intent_f1 = {intent_f1}\n')
            writer.write(f'class_intent_precision = {intent_precision}\n')
            writer.write(f'class_intent_recall = {intent_recall}\n')
            writer.write('\n\n')

            slot_accuracy = accuracy_score(slot_label_all_data, slot_pred_all_data)
            slot_f1 = f1_score(slot_label_all_data, slot_pred_all_data, average='macro')
            slot_precision = precision_score(slot_label_all_data, slot_pred_all_data, average='macro')
            slot_recall = recall_score(slot_label_all_data, slot_pred_all_data, average='macro')

            writer.write(f'slot_accuracy = {slot_accuracy}\n')
            writer.write(f'slot_f1 = {slot_f1}\n')
            writer.write(f'slot_precision = {slot_precision}\n')
            writer.write(f'slot_recall = {slot_recall}\n')

    if do_export:
        def serving_input_fn():
            label_ids = tf.placeholder(tf.int32, [None, num_labels], name='label_ids')
            input_ids = tf.placeholder(tf.int32, [None, max_seq_length], name='input_ids')
            input_mask = tf.placeholder(tf.int32, [None, max_seq_length], name='input_mask')
            segment_ids = tf.placeholder(tf.int32, [None, max_seq_length], name='segment_ids')
            tags_ids = tf.placeholder(tf.int32, [None, max_seq_length], name='tags_ids')
            input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
                'label_ids': label_ids,
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,
                'tags_ids': tags_ids
            })()
            return input_fn

        estimator._export_to_tpu = False
        estimator.export_saved_model(export_dir, serving_input_fn)


if __name__ == "__main__":
    #tf.app.run()
    main()
