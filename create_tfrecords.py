#! /usr/bin/env python

from collections import Counter
from collections import defaultdict
from sklearn.cross_validation import train_test_split
import nltk
import csv
import os
import tensorflow as tf
import pickle
import itertools
import numpy as np

tf.flags.DEFINE_integer(
  "vocab_size", 20000, "Number of words in the vocabulary")

tf.flags.DEFINE_integer(
  "dev_size", 1000, "Size of the dev set")

tf.flags.DEFINE_string(
  "out_path", "./data/processed", "Path to save data and vocab")

FLAGS = tf.flags.FLAGS

OUT_DIR = os.path.abspath(FLAGS.out_path)
if not os.path.exists(OUT_DIR):
  os.makedirs(OUT_DIR)

VOCAB_PATH = os.path.join(FLAGS.out_path, "vocab")
TRAIN_PATH = os.path.join(FLAGS.out_path, "train.tfrecords")
DEV_PATH = os.path.join(FLAGS.out_path, "dev.tfrecords")

def load_reddit_data():
  """
  Returns an iterator over sentences in the data.
  Wraps the sentence into special SENTENCE_START and SENTENCE_END tokens.
  """
  with open("./data/reddit_2008_01.txt") as f:
    reader = csv.reader(f)
    for row in reader:
      yield "SENTENCE_START {} SENTENCE_END".format(row[0].lower())


def create_vocabulary(sequence_iter, vocab_size=None, unk_token="_UNK_", pad_token="_PAD_"):
  print("Building vocabulary... ", end="")
  # Count tokens
  token_counts = Counter()
  for sequence in sequence_iter:
      token_counts.update(sequence)

  # Sort vocabulary by counts. If there is a tie, use alphabetic order
  sorted_vocab = sorted(token_counts.items(), key=lambda _: (_[1], _[0]), reverse=True)
  vocab_items = [_[0] for _ in sorted_vocab[:vocab_size - 1]]
  vocab_items = [pad_token, unk_token] + vocab_items

  # Create dictionaries mapping from token -> index and index -> token
  vocab = defaultdict(lambda: 1, ((token, i) for i, token in enumerate(vocab_items)))

  print("done")
  return vocab


def make_dataset(sequence_iter, vocab):
  print("Transforming sequences...")
  for sequence in sequence_iter:
      x = [vocab[token] for token in sequence]
      y = [vocab[token] for token in sequence[1:]]
      yield x, y
  print("Transforming sequences done.")


def create_tensorflow_sequence_example(x, y):
  ex = tf.train.SequenceExample()
  ex.context.feature["length"].int64_list.value.append(len(x))
  fl_tokens = ex.feature_lists.feature_list["tokens"]
  fl_labels = ex.feature_lists.feature_list["labels"]
  for token in x:
      fl_tokens.feature.add().int64_list.value.append(token)
  for token in y:
      fl_labels.feature.add().int64_list.value.append(token)
  return ex


def create_tfrecords_file(path, sequence_examples):
  path = os.path.abspath(path)
  writer = tf.python_io.TFRecordWriter(path)
  print("Creating TFRecords file at {}...".format(path))
  for example in sequence_examples:
    writer.write(example.SerializeToString())
  writer.close()
  print("Wrote {}".format(path))


if __name__ == "__main__":
  raw_data = list(load_reddit_data())
  data_tokenized = [nltk.word_tokenize(s) for s in raw_data]

  train_tokenized, dev_tokenized = train_test_split(
    data_tokenized,
    test_size=1000)

  vocab = create_vocabulary(
    train_tokenized,
    vocab_size=FLAGS.vocab_size)
  np.save(VOCAB_PATH, list(vocab.items()))
  print("Saved vocabulary to {}".format(VOCAB_PATH))

  ds_iter_train = make_dataset(train_tokenized, vocab)
  ds_iter_dev = make_dataset(dev_tokenized, vocab)

  create_tfrecords_file(
    TRAIN_PATH,
    (create_tensorflow_sequence_example(*s) for s in ds_iter_train))

  create_tfrecords_file(
    DEV_PATH,
    (create_tensorflow_sequence_example(*s) for s in ds_iter_dev))