import tensorflow as tf 

def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask -= 1
  return input_image, input_mask

def load_image(datapoint):
  input_image = tf.image.resize(datapoint[0], (128, 128))
  input_mask = tf.image.resize(datapoint[1], (128, 128))

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask