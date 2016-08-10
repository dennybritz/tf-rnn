import tensorflow as tf
import numpy as np

class SentenceSampleMonitor(tf.contrib.learn.monitors.EveryN):
    def __init__(self, vocab, every_n_steps=100, first_n_steps=1):
        super(SentenceSampleMonitor, self).__init__(
            every_n_steps=every_n_steps,
            first_n_steps=first_n_steps)
        self._vocab = vocab
    def every_n_step_end(self, step, outputs):
        return sample_from_estimator(self._estimator, self._vocab)

def sample_from_estimator(
  estimator,
  vocab,
  max_sentence_len=10,
  start_sentence=["SETENCE_START"]):
  def make_feed_dict_for_sentence(pl, sent):
    joined_sent = " ".join(sent)
    x = np.array(list(vocab.transform([joined_sent])))
    x_len = len(sent)
    return { pl["x"]: x , pl["x_len"]: [x_len] }

  checkpoint_path = tf.train.latest_checkpoint(estimator._model_dir)
  if checkpoint_path is None:
      print("No checkpoint found, not sampling from model.")
      return False
  print("Sampling from model at {}".format(checkpoint_path))

  with tf.Graph().as_default():
    with tf.Session() as sess:

      # Create placeholder variables to feed
      x_pl = {
        "x": tf.placeholder(tf.int64, shape=[None, 40]),
        "x_len": tf.placeholder(tf.int64, shape=[None])
      }

      # Get predictions tensor
      probs_tensor, _, _ = estimator._model_fn(
        x_pl, None, mode=tf.contrib.learn.ModeKeys.INFER)

      # Restore from checkpoint
      saver = tf.train.Saver()
      saver.restore(sess, checkpoint_path)

      # Sample until we reach end of sentence or max length
      sentence = start_sentence
      for i in range(max_sentence_len):
          print(" ".join(sentence))
          feed_dict = make_feed_dict_for_sentence(x_pl, sentence)
          probs = sess.run(probs_tensor, feed_dict=feed_dict)
          next_word_idx = np.random.choice(len(vocab.vocabulary_), p=probs[0][len(sentence)])
          next_word = next(vocab.reverse([[next_word_idx]]))
          sentence.append(next_word)
          if next_word == "EOS":
              break
          if i == max_sentence_len:
              break

def create_lm(vocab_size, embedding_dim, rnn_fn):
  def model_fn(x_dict, y_batch, mode):
    # Unpack input data
    x_batch = x_dict["x"]
    x_batch_len = x_dict["x_len"]
    batch_size = x_batch.get_shape().as_list()[0]

    # Summarize the sequence lengths in tensorboard
    tf.histogram_summary("seq_lengths", x_batch_len)

    # Embed the input words
    with tf.variable_scope("embedding"):
      W = tf.get_variable(
        name="W",
        initializer=tf.random_normal_initializer(-0.05, 0.05),
        shape=[vocab_size, embedding_dim])
      x_embedded = tf.nn.embedding_lookup(W, x_batch)

    # Run the embedded words through an RNN
    with tf.variable_scope("rnn"):
      outputs, last_state = rnn_fn(x_embedded, x_batch_len)

    # Predict the next words
    with tf.variable_scope("outputs"):
      output_dim = outputs[0].get_shape()[1]
      W = tf.get_variable(
        name="W",
        initializer=tf.contrib.layers.xavier_initializer(),
        shape=[output_dim, vocab_size])
      b = tf.get_variable(
        name="b",
        initializer=tf.zeros_initializer(vocab_size))

      # For each time step except the last, calculate logits and probabilities
      # TODO: Can we do this without iteration?
      logit_list = []
      probs_list = []
      for t, o in enumerate(outputs[:-1]):
        logits = tf.nn.xw_plus_b(o, W, b, name="logits")
        probs = tf.nn.softmax(logits, name="probs")
        logit_list.append(logits)
        probs_list.append(probs)

      # Pack logits and probabilities into one tensor.
      # Shape: [batch_size, T-1, num_classes]
      logits = tf.pack(logit_list, axis=1)
      probs = tf.pack(probs_list, axis=1)

    if mode == tf.contrib.learn.ModeKeys.INFER:
      return probs, None, None

    with tf.variable_scope("loss"):
      losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y_batch)
      # We only want to consider the losses that are < sequence length
      # We create a mask that sets all other losses to 0
      tiled_idx = tf.tile(
        tf.expand_dims(tf.range(len(outputs) - 1), 0),
        [batch_size, 1])
      tiled_idx = tf.to_int64(tiled_idx)
      mask = tf.to_float(
        tf.less(
          tiled_idx,
          tf.expand_dims(x_batch_len, 1)))
      # Summary to keep track of non-zero fraction
      summ = tf.scalar_summary('loss_mask_zero_fraction', tf.nn.zero_fraction(mask))
      filtered_losses = losses * mask

      # Mean by actual sequence length (i.e. sum of sequence length)
      mean_loss =  tf.reduce_sum(filtered_losses) / tf.reduce_sum(tf.to_float(x_batch_len) - 1.0)

    if mode == tf.contrib.learn.ModeKeys.TRAIN:
      train_op = tf.contrib.layers.optimize_loss(
          loss=mean_loss,
          global_step=tf.contrib.framework.get_global_step(),
          learning_rate=0.001,
          clip_gradients=10.0,
          optimizer="Adam")
      return probs, mean_loss, train_op
    else:
      return probs, mean_loss, None
  return model_fn