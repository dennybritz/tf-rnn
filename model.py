import tensorflow as tf

def create_lm(vocab_size, embedding_dim, rnn_fn):
  def model_fn(x_dict, y_batch, mode):
    # Unpack input data
    x_batch = x_dict["x"]
    x_batch_len = x_dict["x_len"]
    y_list = tf.unpack(y_batch,  axis=1)
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
      mean_loss =  tf.reduce_mean(filtered_losses)

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