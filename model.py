import tensorflow as tf
import numpy as np

class SentenceSampleMonitor(tf.contrib.learn.monitors.EveryN):
    def __init__(self, vocab, max_sample_length=10, num_samples=5, every_n_steps=100, first_n_steps=1):
        super(SentenceSampleMonitor, self).__init__(
            every_n_steps=every_n_steps,
            first_n_steps=first_n_steps)
        self.vocab = vocab
        self.max_sample_length = max_sample_length
        self.num_samples = num_samples

    def every_n_step_end(self, step, outputs):
        return sample_from_estimator(
          estimator=self._estimator,
          vocab=self.vocab,
          max_sample_length=self.max_sample_length,
          num_samples=self.num_samples)


def sample_from_estimator(
  estimator,
  vocab,
  max_sample_length=10,
  num_samples=5,
  start_sentence=["SENTENCE_START"]):

  def make_feed_dict_for_sentence(pl, sent):
    x = np.array([vocab[_] for _ in sent])
    x_len = len(x)
    return { pl["x"]: x , pl["x_len"]: [x_len] }

  vocab_inverse = {v: k for k, v in vocab.items()}
  checkpoint_path = tf.train.latest_checkpoint(estimator._model_dir)
  if checkpoint_path is None:
      print("No checkpoint found, not sampling from model.")
      return False
  print("Sampling from model at {}".format(checkpoint_path))

  with tf.Graph().as_default():
    with tf.Session() as sess:

      # Create placeholder variables to feed
      x_pl = {
        "x": tf.placeholder(tf.int64, shape=[None, vocab.max_document_length]),
        "x_len": tf.placeholder(tf.int64, shape=[None])
      }

      # Get predictions tensor
      probs_tensor, _, _ = estimator._model_fn(
        x_pl, None, mode=tf.contrib.learn.ModeKeys.INFER)

      # Restore from checkpoint
      saver = tf.train.Saver()
      saver.restore(sess, checkpoint_path)

      # Sample until we reach end of sentence or max length
      for _ in range(num_samples):
        sentence = []
        word_probs = []
        sentence.append(*start_sentence)
        for i in range(max_sample_length):
            feed_dict = make_feed_dict_for_sentence(x_pl, sentence)
            probs = sess.run(probs_tensor, feed_dict=feed_dict)
            next_word_idx = np.random.choice(
              len(vocab.vocabulary_),
              p=probs[0][len(sentence) - 1])
            next_word = vocab_inverse[next_word_idx]
            word_probs.append(probs[0][len(sentence) - 1][next_word_idx])
            sentence.append(next_word)
            if next_word == "SENTENCE_END":
                break
            if i == max_sample_length:
                break
        print(" ".join(sentence))
        print(word_probs)


def create_language_model_rnn(vocab_size, embedding_dim, max_sequence_len):
  def model_fn(x_dict, y_batch, mode):

    # We don't make a prediction for the last token since it's always "End of Sentnece"
    x_batch = x_dict["tokens"]
    x_batch = tf.slice(x_batch, [0,0], tf.shape(y_batch))

    x_batch_len = x_dict["length"]
    x_batch_len = tf.minimum(x_batch_len, max_sequence_len)

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
      cell = tf.nn.rnn_cell.GRUCell(128)
      outputs, last_state = tf.nn.dynamic_rnn(
        cell,
        x_embedded,
        dtype=tf.float32,
        sequence_length=x_batch_len)

    # Calculate the logits and the predictions
    with tf.variable_scope("logits"):
      W = tf.get_variable(
        name="W",
        initializer=tf.contrib.layers.xavier_initializer(),
        shape=[cell.output_size, vocab_size])
      b = tf.get_variable(
        name="b",
        initializer=tf.zeros_initializer(vocab_size))

      # For each time step, calculate the logits for the batch
      outputs_by_time = tf.transpose(outputs, [1,0,2])
      logits = tf.map_fn(lambda x_t: tf.batch_matmul(x_t, W), outputs_by_time, name="logits")
      probs = tf.map_fn(tf.nn.softmax, logits, name="probs")

    with tf.variable_scope("loss"):
      # For each time step, calculate the batch losses
      targets_by_time = tf.transpose(y_batch)
      # Ignore all losses where y=0 due to the padding
      loss_weight_mask = tf.sign(tf.to_float(targets_by_time))

      # Calculate the losses for each timestep. Final Shape: [T, B]
      losses = tf.map_fn(
        lambda _: _[2] * tf.nn.sparse_softmax_cross_entropy_with_logits(_[0], _[1], name="loss_by_t"),
        [logits, targets_by_time, loss_weight_mask],
        dtype=tf.float32,
        name="losses")

      # Calculate the mean loss
      # We divide the "actual" number of prediction we made
      mean_loss = tf.reduce_sum(losses) / tf.reduce_sum(loss_weight_mask)

    if mode == tf.contrib.learn.ModeKeys.INFER:
      return probs, None, None
    elif mode == tf.contrib.learn.ModeKeys.TRAIN:
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