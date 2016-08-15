import tensorflow as tf

from collections import Counter
from collections import defaultdict
from collections import namedtuple
from sklearn.cross_validation import train_test_split
import nltk
import csv
import os
import tensorflow as tf
import pickle
import itertools
import numpy as np

def parse_sequence_example(example):
  context_features = {
    "length": tf.FixedLenFeature([], dtype=tf.int64)
  }
  sequence_features = {
    "tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64),
    "labels": tf.FixedLenSequenceFeature([], dtype=tf.int64)
  }

  context, sequence = tf.parse_single_sequence_example(
    serialized=example,
    context_features=context_features,
    sequence_features=sequence_features,
    example_name="sequence_example"
  )

  return dict(x.items() | y.items())

def create_tf_input_fn(filenames, batch_size, num_epochs=None):
  def input_fn():
    filename_q = tf.train.string_input_producer(
        filenames,
        num_epochs=num_epochs)

    # Read from the file and decode the examples
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_q)
    example = parse_sequence_example(serialized_example)

    # Batch and pad examples
    example_batch = tf.train.batch(
          tensors=example,
          batch_size=batch_size,
          capacity=5000 + batch_size * 10,
          dynamic_pad=True)

    return {
        "tokens": example_batch["tokens"],
        "length": example_batch["length"]
    }, example_batch["labels"]

  return input_fn


def create_reddit_inputs(path, batch_size, train_epochs):
  Dataset = namedtuple("Dataset",
    ["vocab", "train_input_fn", "dev_input_fn"])

  vocab_path = os.path.join(path, "vocab.npy")
  train_tfrecords_path = os.path.join(path, "train.tfrecords")
  dev_tfrecords_path = os.path.join(path, "dev.tfrecords")

  # Load vocabulary
  vocab = dict(np.load(vocab_path))
  train_input_fn = create_tf_input_fn([train_tfrecords_path], batch_size, train_epochs)
  dev_input_fn = create_tf_input_fn([dev_tfrecords_path], batch_size, 1)

  return Dataset(
    vocab=vocab,
    train_input_fn=train_input_fn,
    dev_input_fn=dev_input_fn
  )