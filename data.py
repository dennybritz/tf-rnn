from sklearn.cross_validation import train_test_split
from collections import namedtuple
import tensorflow as tf
import numpy as np
import csv

Dataset = namedtuple("Dataset", [
  "x_train", "x_len_train", "y_train",
  "x_dev", "x_len_dev", "y_dev",
  "vocab"])

def load_reddit_data(
  max_document_length=25,
  min_frequency=1000,
  test_size=10000):

  # Load Data into memory
  with open("./data/reddit_2008_01.txt") as f:
      reader = csv.reader(f)
      data_raw = [x[0] + " EOS" for x in reader]

  # Preprocess data
  vocab = tf.contrib.learn.preprocessing.text.VocabularyProcessor(
    max_document_length=max_document_length,
    min_frequency=min_frequency)
  vocab.fit(data_raw)

  # Create numpy arrays from the data
  x = np.array(list(vocab.transform(data_raw)))
  x_lengths = np.array([len(_) for _ in vocab._tokenizer(data_raw)])
  y = np.array([_[1:] for _ in x])

  # Split into train/dev data
  x_train, x_dev, x_len_train, x_len_dev, y_train, y_dev = train_test_split(
    x, x_lengths, y ,test_size=test_size)

  return Dataset(
    x_train=x_train,
    x_len_train=x_len_train,
    y_train=y_train,
    x_dev=x_dev,
    x_len_dev=x_len_dev,
    y_dev=y_dev,
    vocab=vocab
  )

def print_dataset_stats(ds):
  print("Vocabulary Size: {}".format(len(ds.vocab.vocabulary_)))
  print("Train Data Shape: {}".format(ds.x_train.shape))
  print("Dev Data Shape {}".format(ds.x_dev.shape))


def create_tf_input_fn(x, x_len, y, batch_size=32, num_epochs=None):
  def input_fn():
    x_slice, x_len_slice, y_slice = tf.train.slice_input_producer(
      [x, x_len, y],
      num_epochs=num_epochs,
      shuffle=True,
      capacity=50000 + batch_size * 10)
    x_batch, x_batch_len, y_batch = tf.train.batch(
      [x_slice, x_len_slice, y_slice],
      batch_size=batch_size)
    return { "x": x_batch, "x_len": x_batch_len}, y_batch
  return input_fn

def create_train_dev_input_fns(ds, batch_size=32):
  train_input_fn = create_tf_input_fn(
    ds.x_train, ds.x_len_train, ds.y_train, batch_size)
  dev_input_fn = create_tf_input_fn(
    ds.x_dev, ds.x_len_dev, ds.y_dev, batch_size, 1)
  return train_input_fn, dev_input_fn