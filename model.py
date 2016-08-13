import tensorflow as tf
import numpy as np

class SentenceSampleMonitor(tf.contrib.learn.monitors.EveryN):
    def __init__(self, vocab, max_sample_length=10, num_samples=5, every_n_steps=100, first_n_steps=1):
        super(SentenceSampleMonitor, self).__init__(
            every_n_steps=every_n_steps,
            first_n_steps=first_n_steps)
        self._vocab = vocab
        self.max_sample_length = max_sample_length
        self.num_samples = num_samples

    def every_n_step_end(self, step, outputs):
        return sample_from_estimator(
          estimator=self._estimator,
          vocab=self._vocab,
          max_sample_length=self.max_sample_length,
          num_samples=self.num_samples)


def sample_from_estimator(
  estimator,
  vocab,
  max_sample_length=10,
  num_samples=5,
  start_sentence=["SENTENCE_START"]):

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
            next_word = next(vocab.reverse([[next_word_idx]]))
            word_probs.append(probs[0][len(sentence) - 1][next_word_idx])
            sentence.append(next_word)
            if next_word == "SENTENCE_END":
                break
            if i == max_sample_length:
                break
        print(" ".join(sentence))
        print(word_probs)


def create_language_model_rnn(vocab_size, embedding_dim, rnn_fn):
  def model_fn(x_dict, y_batch, mode):
    # Unpack input data
    x_batch = x_dict["x"]
    x_batch_len = tf.to_int32(x_dict["x_len"])
    batch_size = x_batch.get_shape().as_list()[0]

    max_sequence_len = x_batch.get_shape().as_list()[1]
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

      # For each example, only consider the losses within the example sequence length
      losses = []
      logit_by_ex = tf.unpack(logits)
      target_by_ex = tf.unpack(y_batch)
      sequence_len_by_ex = tf.unpack(x_batch_len - tf.ones_like(x_batch_len))

      # For each example in the batch...
      for example_logits, example_length, example_targets in zip(logit_by_ex, sequence_len_by_ex, target_by_ex):

        # Slice logits to the length of the example
        logits_slice = tf.slice(
          example_logits,
          begin=[0, 0],
          size=[1, 0] * (example_length) + [0, -1],
          name="logits_slice")

        # Slice targets (y) to the length of the example
        target_slice = tf.slice(
          example_targets,
          begin=[0],
          size=[1] * example_length,
          name="target_slice")

        # Mean loss for this example
        example_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits_slice, target_slice, name="example_losses")
        example_mean_loss = tf.reduce_mean(example_losses, name="example_mean_loss")
        losses.append(example_mean_loss)

      # Average Loss over the beach
      mean_loss = tf.reduce_mean(losses, name="mean_loss")

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